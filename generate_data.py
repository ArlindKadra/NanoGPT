import os

import torch

from models.bigram import BigramLanguageModel

batch_size = 64
n_embed = 384
context_size = 256
eval_iterations = 500
max_iterations = 5000
dropout = 0.2
learning_rate = 3e-4
nr_blocks = 6
nr_heads = 8

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

model = BigramLanguageModel(n_embed, vocab_size, context_size, nr_blocks=nr_blocks, nr_heads=nr_heads, dropout=dropout).to(device)
model.load_state_dict(torch.load('gpt.pt'))
model.eval()

context = torch.zeros(1, 1, dtype=torch.long, device=device)
generated_tokens = model.generate(context, 1000)[0]
generated_text = decode(generated_tokens.tolist())

with open('generated_text.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(generated_text))