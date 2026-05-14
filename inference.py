import torch
from model import GPTLanguageModel  # or your NanoGPT model
from tokenizer import decode, chars, vocab_size

# -------------------
# device
# -------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# model
# -------------------

model = GPTLanguageModel(vocab_size)
model.load_state_dict(torch.load("nanogpt_model.pth", map_location=device))

model = model.to(device)
model.eval()

# -------------------
# generate
# -------------------

context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated = model.generate(context, max_new_tokens=300)

output = decode(generated[0].tolist())

print(output)