"""Optional integration check: actual NVIDIA model code, tiny random weights.

Downloads configuration/model Python code only, never pretrained weights.
Run explicitly; trust_remote_code executes code from the selected NVIDIA repo.
"""

import torch
from transformers import AutoConfig, AutoModel

from maskdecide.decoding import decode
from maskdecide.slot_attention import install_slot_attention, verify_model_isolation


def main():
    torch.manual_seed(17)
    repository = "nvidia/Nemotron-Labs-Diffusion-3B"
    # Revision inspected when building this integration. Keep the API's model
    # choice independent; this script is a reproducible architecture test.
    revision = "0d51902da1f8869f83413ce642fab402fa5641e0"
    config = AutoConfig.from_pretrained(repository, revision=revision, trust_remote_code=True)
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.vocab_size = 256
    config.mask_token_id = 255
    config._attn_implementation = "sdpa"
    model = AutoModel.from_config(config, trust_remote_code=True, dtype=torch.float32).eval()
    install_slot_attention(model)
    print("Actual NVIDIA implementation, reduced dimensions, random weights:")
    print(verify_model_isolation(model, [1, 2, 3, 4, 5, 6], 255))
    tokens = torch.tensor([[1, 2, 3, 4, 255, 5, 6, 255]])
    groups = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])
    for mode in ("one_pass", "iterative"):
        result = decode(model, tokens, [4, 7], [[1, 2], [3, 4, 5]],
                        mask_id=255, mode=mode, group_ids=groups)
        assert len(result.winners) == 2
        assert [len(p) for p in result.probabilities] == [2, 3]
        print(mode, "passes=", result.forward_passes, "OK")


if __name__ == "__main__":
    main()
