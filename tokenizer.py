with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[ch] for ch in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

if __name__ == "__main__":
    data = encode("hello")

    print(data[:20])
    print(decode(data[:20]))