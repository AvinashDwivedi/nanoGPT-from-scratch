from tqdm import tqdm
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

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(200)

        for k in range(200):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out

max_iter = 50000
eval_interval = 500

pbar = tqdm(range(max_iter), desc="Training")
for steps in pbar:
    if (steps % eval_interval == 0) or (steps == max_iter - 1):
        losses = estimate_loss()
        pbar.set_postfix({
            "train_loss": f"{losses['train']:.4f}",
            "val_loss": f"{losses['val']:.4f}"
        })
        torch.save(model.state_dict(), "nanogpt_model.pth")

    xb, yb = get_batch("train")
    xb = xb.to(device)
    yb = yb.to(device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long).to(device)

print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))