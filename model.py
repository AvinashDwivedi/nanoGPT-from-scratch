import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig

# hyperparameters
batch_size = 32
block_size = 16
n_embd = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

class Head(nn.Module):
    def __init__(self, head_size, dropout=0.2):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        # causal mask
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # compute attention scores
        wei = q @ k.transpose(-2, -1)

        # scaling
        wei = wei * (C ** -0.5)

        # causal masking
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # normalize
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)

        out = wei @ v

        return out
    
class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd
        )

        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, dropout=0.2):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(head_size * num_heads, n_embd)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()

        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(
        num_heads, head_size
        )

        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            Block(n_embd, num_heads=16),
            Block(n_embd, num_heads=16),
            Block(n_embd, num_heads=16),
            Block(n_embd, num_heads=16),
            Block(n_embd, num_heads=16),
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):

        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )
        x = token_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=0.8, k=20):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]
            v, ix = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float("inf")
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx