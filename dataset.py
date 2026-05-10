import torch
from tokenizer import text, stoi, itos, encode, decode

data = torch.tensor(encode(text), dtype=torch.long)

# Train and test splits
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
batch_size = 4

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