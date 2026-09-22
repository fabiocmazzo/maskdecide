# Third-party components

MaskDecide's original code and documentation are licensed under the Apache
License 2.0. This license does not replace the terms of any model or dependency.

## Fast-dLLM v2 1.5B

MaskDecide loads `Efficient-Large-Model/Fast_dLLM_v2_1.5B` from Hugging Face.
Model weights, tokenizer assets, and model implementation files are downloaded
separately and are not bundled in this repository.

- [Model repository and model card](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B)
- [Model metadata](https://huggingface.co/api/models/Efficient-Large-Model/Fast_dLLM_v2_1.5B)
- [Upstream Fast-dLLM project](https://github.com/NVlabs/Fast-dLLM)
- [Upstream code license](https://github.com/NVlabs/Fast-dLLM/blob/main/LICENSE)
- [Fast-dLLM v2 paper](https://arxiv.org/abs/2509.26328)

The model card metadata declares `apache-2.0`, verified on September 21, 2026
at revision `25093b6f63300adfd57f72145083c8a528fe4f16`. It identifies
`Qwen/Qwen2.5-1.5B-Instruct` as the base model. The upstream model's license,
attribution requirements, and any applicable base-model terms remain in effect.
Consult the upstream repositories when using or redistributing those assets.
This record identifies the metadata checked; it does not pin the runtime download.

Fast-dLLM v2 is the work of Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao,
Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie.

## Python dependencies

Dependencies retain their respective licenses. Their resolved versions and
sources are recorded in `uv.lock`; they are not relicensed by MaskDecide.

## Jev

Jev is a product of TypeSafe AI. MaskDecide includes an independently implemented,
experimental adapter for its HTTP request and response format. No Jev model
weights are included. MaskDecide is not affiliated with or endorsed by TypeSafe AI.
