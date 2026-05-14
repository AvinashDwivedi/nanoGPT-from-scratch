from dataclasses import dataclass

@dataclass
class GPTConfig:

    vocab_size: int = 50257
    block_size: int = 256

    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

    dropout: float = 0.2