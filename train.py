import torch
from model import GPTLanguageModel
from dataset import get_batch
from tokenizer import decode, chars


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

model = GPTLanguageModel(len(chars))
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 4

for steps in range(10000):
    xb, yb = get_batch("train")
    xb = xb.to(device)
    yb = yb.to(device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % 100 == 0:
        print(f"step {steps}: loss {loss.item()}")


context = torch.zeros((1, 1), dtype=torch.long).to(device)

print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))