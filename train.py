import os

import tiktoken
import torch

from models.bigram import BigramLanguageModel


torch.manual_seed(11)
# data path
data_path = os.path.expanduser(
    os.path.join(
        '~',
        'Desktop',
    )
)

file_path = os.path.join(data_path, 'tinyshakespeare.txt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read data
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#enc = tiktoken.encoding_for_model("gpt-4")
#tokens = torch.tensor(enc.encode(text), dtype=torch.long)
tokens = torch.tensor(encode(text), dtype=torch.long)
# split data
train_data = tokens[:int(len(tokens) * 0.9)]
valid_data = tokens[int(len(tokens) * 0.9):]

batch_size = 64
n_embed = 384
context_size = 256
eval_iterations = 500
max_iterations = 5000
dropout = 0.2
learning_rate = 3e-4
nr_blocks = 6
nr_heads = 8
"""
x = tokens[:context_size]
y = tokens[1:context_size + 1]
for t in range(context_size):
    input_data = x[:t + 1]
    target = y[t]
    print(input_data)
    print(target)
"""

def get_batch(split):

    if split == 'train':
        data = train_data
    else:
        data = valid_data

    start_ixs = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[start_ix:start_ix + context_size] for start_ix in start_ixs])
    y = torch.stack([data[start_ix + 1:start_ix + 1 + context_size] for start_ix in start_ixs])
    x = x.to(device)
    y = y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss(model):

    model.eval()
    out = {}
    for split in ['train', 'valid']:
        loss = 0.0
        for _ in range(eval_iterations):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            loss += loss.item()
        loss /= eval_iterations
        out[split] = loss
    model.train()

    return out

model = BigramLanguageModel(n_embed, vocab_size, context_size, nr_blocks=nr_blocks, nr_heads=nr_heads, dropout=dropout).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iteration in range(max_iterations):

    if iteration % eval_iterations == 0:
        out = estimate_loss(model)
        print(f"Iteration {iteration}, Train loss: {out['train']}, Valid loss: {out['valid']}")

    x, y = get_batch('train')
    optimizer.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()

# save model
torch.save(model.state_dict(), 'gpt.pt')

context = torch.zeros(1, 1, dtype=torch.long, device=device)
model.eval()
generated_text = model.generate(context, 400)
print([decode(char_gen) for char_gen in generated_text.tolist()])
