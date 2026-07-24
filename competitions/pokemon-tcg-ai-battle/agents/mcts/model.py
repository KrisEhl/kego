"""Transformer policy/value model for sparse Pokémon TCG observations.

The value path uses an EmbeddingBag sparse input, transformer encoder, and scalar
value head. The policy path uses a decoder EmbeddingBag and cross-attention layers
over the encoded state before its scalar action head. ``MODEL_ARGS`` contains
``(d_model, num_heads, d_feedforward, encoder_layers, decoder_layers)``;
``num_heads`` must divide ``d_model``.
"""

import torch

from cg.api import all_attack, all_card_data


def _load_feature_counts() -> tuple[int, int]:
    try:
        cards = all_card_data()
        return max(cards, key=lambda card: card.cardId).cardId + 1, max(
            all_attack(), key=lambda attack: attack.attackId
        ).attackId + 1
    except Exception:
        return 2000, 2000


CARD_COUNT, ATTACK_COUNT = _load_feature_counts()

NUM_WORDS_ENCODER = 24
ENCODER_SIZE = 22000
DECODER_MAIN_FEATURE = 8
DECODER_ATTACK_OFFSET = 14
DECODER_CARD_OFFSET = DECODER_ATTACK_OFFSET + ATTACK_COUNT
DECODER_SIZE = DECODER_CARD_OFFSET + (1 + DECODER_MAIN_FEATURE + 48) * CARD_COUNT

# (d_model, num_heads, d_feedforward, n_encoder_layers, n_decoder_layers).
# Single source of truth: training and inference both build PolicyValueNet
# from this, so checkpoints always match. num_heads must divide d_model.
MODEL_ARGS = (256, 4, 512, 2, 2)


def ensure_tensors(state_dict: dict) -> dict:
    """Ensure all numpy array values in state_dict are converted to PyTorch Tensors."""
    import numpy as np

    res = {}
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            res[k] = torch.from_numpy(v)
        else:
            res[k] = v
    return res


def _layer_count(state_dict: dict, prefix: str, suffix: str) -> int:
    found = []
    for key in state_dict:
        if key.startswith(prefix) and key.endswith(suffix):
            try:
                found.append(int(key[len(prefix) :].split(".", 1)[0]))
            except ValueError:
                pass
    return max(found) + 1 if found else 0


def model_args_from_state_dict(state_dict: dict) -> tuple[int, int, int, int, int]:
    d_model = int(state_dict["encoder_bag.weight"].shape[1])
    d_feedforward = int(state_dict["encoder.layers.0.linear1.weight"].shape[0])
    num_heads = MODEL_ARGS[1] if d_model % MODEL_ARGS[1] == 0 else 4
    return (
        d_model,
        num_heads,
        d_feedforward,
        _layer_count(state_dict, "encoder.layers.", ".linear1.weight"),
        _layer_count(state_dict, "decoder.", ".fc1.weight"),
    )


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int) -> None:
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = torch.nn.functional.relu(y)
        y = self.fc2(y)
        return self.norm2(res + y)


class PolicyValueNet(torch.nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_feedforward: int, num_layers_encoder: int, num_layers_decoder: int
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder_bag = torch.nn.EmbeddingBag(ENCODER_SIZE, d_model, mode="sum")
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, 0)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers_encoder, enable_nested_tensor=False)
        self.encoder_fc = torch.nn.Linear(d_model, 1)
        self.decoder_bag = torch.nn.EmbeddingBag(DECODER_SIZE, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList()
        for _ in range(num_layers_decoder):
            self.decoder.append(DecoderLayer(d_model, num_heads, d_feedforward))
        self.decoder_fc = torch.nn.Linear(d_model, 1)

    def forward(
        self,
        index_encoder: torch.Tensor,
        value_encoder: torch.Tensor,
        offset_encoder: torch.Tensor,
        index_decoder: torch.Tensor,
        value_decoder: torch.Tensor,
        offset_decoder: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = v.reshape(-1, NUM_WORDS_ENCODER, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        v = torch.tanh(v.mean(0))

        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        p = torch.tanh(p)
        return (v, p)


MyModel = PolicyValueNet
