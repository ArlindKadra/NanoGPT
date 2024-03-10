import torch
import torch.nn as nn

from torch.nn import functional as F


class MultiAttentionHead(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout=0.2):
        super(MultiAttentionHead, self).__init__()

        head_size = int(embedding_size / num_heads)
        self.heads = nn.ModuleList([AttentionHead(embedding_size, head_size, dropout) for _ in range(num_heads)])

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        return out

class FeedForward(nn.Module):
    def __init__(self, embedding_size, dropout=0.2):
        super(FeedForward, self).__init__()

        self.ln_1 = nn.Linear(embedding_size, 4 * embedding_size)
        self.act_func = nn.ReLU()
        self.ln_2 = nn.Linear(4 * embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = self.ln_1(x)
        out = self.act_func(out)
        out = self.ln_2(out)
        out = self.dropout(out)

        return out

class AttentionHead(nn.Module):
    def __init__(self, embedding_size, head_size, dropout):
        super(AttentionHead, self).__init__()

        self.ln_q = nn.Linear(embedding_size, head_size)
        self.ln_k = nn.Linear(embedding_size, head_size)
        self.ln_v = nn.Linear(embedding_size, head_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril_mask', torch.tril(torch.ones(embedding_size, embedding_size)))

    def forward(self, x):

        q = self.ln_q(x)
        k = self.ln_k(x)
        v = self.ln_v(x)

        attention_logits = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        attention_logits.masked_fill_(self.tril_mask[:q.shape[1], :q.shape[1]] == 0, float('-inf'))
        attention = F.softmax(attention_logits, dim=-1)
        attention = self.dropout(attention)
        out = attention @ v

        return out

class AttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout):
        super(AttentionBlock, self).__init__()

        self.multi_head_attention = MultiAttentionHead(embedding_size, num_heads, dropout)
        self.feed_forward = FeedForward(embedding_size, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):

        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))

        return x
class BigramLanguageModel(nn.Module):

    def __init__(self, n_embed, vocab_size, context_size, nr_blocks=4, nr_heads=8, dropout=0.2):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(context_size, n_embed)
        self.attention_blocks = nn.Sequential(*[AttentionBlock(n_embed, nr_heads, dropout) for _ in range(nr_blocks)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.ln_head = nn.Linear(n_embed, vocab_size)
        self.context_size = context_size

    def forward(self, x, y=None):

        token_embeddings = self.token_embedding(x)
        pos_embeddings = self.pos_embedding(torch.arange(x.shape[1], device=x.device))
        x = token_embeddings + pos_embeddings
        x = self.attention_blocks(x)
        x = self.ln_f(x)
        logits = self.ln_head(x)

        if y is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)
            loss = F.cross_entropy(logits, y)
            return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_next = idx[:, -self.context_size:]
            # get the predictions
            logits, loss = self(idx_next)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
