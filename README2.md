# MaskDecide: atenção por pergunta

Pacote de integração para o projeto existente. Contém os três módulos completos
que trabalham juntos:

- `src/maskdecide/api.py`: API e montagem dos grupos de tokens.
- `src/maskdecide/decoding.py`: decoder com projeção restrita `F.linear`.
- `src/maskdecide/slot_attention.py`: backend de atenção e verificação de isolamento.

## Instalação

Copie os três arquivos para `src/maskdecide/` no seu projeto, substituindo os
dois existentes e adicionando `slot_attention.py`. Preserve os demais arquivos,
incluindo o entrypoint, `__init__.py`, dependências e `middlewares/http_logger.py`.
Este pacote é uma integração ao projeto; não é um servidor independente.

Execute como antes:

```bash
uv run maskdecide --host 127.0.0.1 --port 8000
```

## Controles no api.py

```python
MAX_QUESTIONS_PER_PROMPT = 3
USE_SLOT_ATTENTION = True
VERIFY_SLOT_ATTENTION_ON_STARTUP = True
```

Os três controles são do servidor. O campo `isolated` continua aceito por
compatibilidade, mas é ignorado. `JEV_ISOLATED` também não é consultado.
`DECODER_MODE=one_pass` ou `iterative` e `DECODER_THRESHOLD` continuam independentes.

`MAX_QUESTIONS_PER_PROMPT` controla quantas perguntas entram na mesma sequência.
Com três, nove perguntas geram três sequências processadas uma após a outra.
`USE_SLOT_ATTENTION` controla o acesso entre tokens dentro de cada sequência.
Com `True`, mesmo as três perguntas que compartilham o prompt ficam isoladas entre si.

O limite é de perguntas, não de máscaras: uma `choice` com 30 alternativas usa
30 slots pertencentes ao mesmo grupo. `MAX_INPUT_TOKENS` continua valendo por
sequência, incluindo os slots. Não há truncamento nem subdivisão automática por
tokens; uma sequência acima do limite retorna HTTP 422.

## Regra de atenção

Cada token recebe um grupo. Grupo 0 contém estado, instruções compartilhadas e
delimitadores do template. Cada pergunta `noul` ou `choice` recebe um grupo
positivo, incluindo seus critérios, alternativas, rótulos de resposta e todos
os seus slots. Perguntas `score` não usam máscara: ficam no grupo 0, com
visibilidade total nos dois sentidos.

| Tokens que consultam | Conteúdo acessível |
|---|---|
| Estado/instruções compartilhadas e perguntas `score` | Tudo |
| Pergunta 1, critérios e slots | Grupo 0 e grupo 1 |
| Pergunta 2, critérios e slots | Grupo 0 e grupo 2 |

A regra é bidirecional dentro de cada grupo e aplicada em todas as camadas.
O estado mantém visibilidade total: escondê-lo das perguntas corrompia as
representações das quais todos os slots dependem. O isolamento entre perguntas
é direto (nenhuma camada atende de uma pergunta a outra); pode existir
comunicação indireta residual através do estado, inerente a qualquer máscara
assimétrica em múltiplas camadas.

O texto do prompt mantém a organização anterior: perguntas primeiro, slots no
final. Os grupos são associados pelo intervalo de caracteres construído pela
API e pelos offsets do tokenizer; não por buscas textuais de `Question 1` no
conteúdo do usuário. É necessário um tokenizer rápido (`is_fast=True`).

## Por que há um backend adicional?

Na implementação publicada e inspecionada do Nemotron, o encoder descarta a
máscara normal quando `use_causal_mask=False`, e a camada de difusão entrega
`None` ao backend. Apenas acrescentar `attention_mask=...` ao forward não basta.

A integração registra um backend próprio no registro de attention usado pelo
modelo e troca somente a seleção de backend das suas camadas `Ministral3Attention`.
As configurações do encoder e do modelo principal continuam preservadas.
Q/K/V, RoPE, escalonamento das queries, projeções e pesos quantizados continuam
sendo calculados pelo código original. O backend aplica SDPA com nossa máscara.
Não é necessário editar arquivos do cache Hugging Face.

O contexto da máscara dura um único forward e é limpo mesmo em caso de exceção.
Cada forward verifica que todas as camadas aplicaram a máscara exatamente uma vez.
Uma classe de attention não suportada ou um caminho que ignore a máscara gera erro.
Não existe fallback silencioso para atenção global quando o modo estruturado está ativo.

## Decoder

A projeção continua calculando somente as linhas necessárias da cabeça de saída.
Logits/probabilidades são capturados antes de preencher cada máscara. O tensor
original de entrada não é modificado. O novo argumento opcional é `group_ids`;
sem ele, o caminho de decodificação original é mantido, inclusive em `/decide`.

No modo iterativo estruturado, o scheduler garante progresso por pergunta:
aceita os slots confiantes daquela pergunta ou, se nenhum passou do threshold,
aceita o mais confiante dela. Assim uma pergunta não espera o progresso de outra.
Para perguntas com apenas um slot, isso normalmente resulta em um único forward
mesmo no modo iterativo. Múltiplos slots da mesma pergunta ainda podem exigir
várias passagens. Isso não é refinamento de respostas já comprometidas.

## Validação

Na inicialização, a verificação opcional faz três forwards pequenos no modelo
real carregado. Ela mantém comprimento e posições fixos, troca a pergunta 2 e
verifica que os tokens do estado se movem (o estado enxerga todas as
perguntas). Depois troca o estado e verifica que os dois slots mudam.
O resultado aparece como `SLOT ATTENTION PROBE`.
Além disso, a contagem de camadas é verificada em todos os forwards estruturados.

Testes locais sem baixar pesos:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -p 'test_slot_attention.py' -v
```

Teste opcional com o código real da NVIDIA e uma configuração reduzida de duas
camadas, inicializada com pesos aleatórios:

```bash
PYTHONPATH=src uv run python tests/verify_nemotron_cpu.py
```

Esse último teste baixa apenas configuração/código de uma revisão fixada,
executando o código remoto da NVIDIA com `trust_remote_code=True`. Não baixa os
pesos treinados. Aqui passaram os nove testes de regressão e o teste de integração
com a implementação real reduzida, em CPU com PyTorch 2.14.0 e Transformers 5.17.0.
O teste real mediu diferença máxima 0.0 nas representações protegidas ao alterar
a outra pergunta. O checkpoint completo quantizado não foi executado aqui.

## Comparação de qualidade e limites

Para comparar, mantenha os mesmos exemplos, tamanho de grupo e decoder; altere
somente `USE_SLOT_ATTENTION` e reinicie o servidor. O caminho desativado preserva
os mesmos tokens de entrada. Em `iterative`, a ativação também altera o scheduler;
use `one_pass` primeiro para isolar melhor o efeito da máscara.

Isolamento estrutural não garante melhor F1. O estado agora é codificado sem
consultar as perguntas; mesmo com uma pergunta, isso difere da atenção global
original. As posições RoPE e o template também continuam influenciando o modelo.

Esta implementação é um protótipo de inferência, com uma sequência sem padding,
sem KV cache, sem `torch.compile`, sem treinamento e sem paralelismo de modelo.
A máscara booleana é densa, de tamanho L²; não há promessa de ganho de velocidade
ou VRAM. O kernel escolhido por SDPA pode mudar. Não substitua por FlexAttention
sem uma comparação posterior de comportamento e desempenho.

Fontes usadas para verificar o caminho de integração:

- https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/blob/0d51902da1f8869f83413ce642fab402fa5641e0/modeling_ministral.py
- https://huggingface.co/docs/transformers/main/en/attention_interface
- https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
