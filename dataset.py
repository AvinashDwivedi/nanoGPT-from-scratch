import torch
from tokenizer import encode

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()


data = torch.tensor(encode(text), dtype=torch.long)

# Train and test splits
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 16
batch_size = 256

def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size, (batch_size, ))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

if __name__ == "__main__":
    x, y = get_batch("train")
    print(x.shape)
    print(x)
    print(y.shape)
    print(y)