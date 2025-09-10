import torch
import torch.nn as nn
import torch.nn.functional as F
from data_preprocessing import vocab_size, device

# Model Parameters:
n_embd = 256
head_size = 32
n_layer = 8
n_head = 8
batch_size = 32
block_size = 27
dropout = 0.1

# Attention Head:
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        # K,Q,V Matrices:
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Buffer Matrix and Dropout Layer:
        self.register_buffer('tril', torch.tril(torch.ones([block_size, block_size])))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5

        # Padding Masking:
        is_padding = (x == 0).all(dim=-1)  # Shape: [B, T]
        padding_mask = is_padding.unsqueeze(1).expand(-1, T, -1)  # Shape: [B, T, T]
        wei = wei.masked_fill(padding_mask, float('-inf'))

        # Causal Masking:
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Adjust Embeddings:
        v = self.value(x)
        out = wei @ v
        return out

# Multi-Headed Attention:
class MultiHeadedAttention(nn.Module):

    def __init__(self, head_size, n_head):
        super().__init__()

        # List of Attention Heads:
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])

        # Projection and Dropout Layers:
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Multi-Layer Perceptron:
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        # Linear Layers, ReLU and Dropout:
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Self-Attention/FeedForward Block:
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head

        # Self-Attention/FeedForward:
        self.sa = MultiHeadedAttention(head_size, n_head)
        self.ffwd = FeedForward(n_embd)

        # Layer Normalization:
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    # Residual Blocks and Self-Attention/FeedForward:
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# AMP Transformer Model:
class AMPTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        # Token and Positional Embedding Tables:
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Block Layers:
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        # Layer Normalization and Unembedding:
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Regression Head:
        self.mic_head = nn.Linear(n_embd, 1)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # Embeddings:
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        # Creating Logits after Forward Pass:
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Determining Loss via Cross Entropy:
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=22)

        return logits, loss

    def get_representation(self, idx):
        B,T = idx.shape
            
        # Embeddings:
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x[:, 0, :] 
            
        return x

    def predict_mic(self, idx):

        # Predict MIC Value with Regression Head:
        representation = self.get_representation(idx)
        mic_value = self.mic_head(representation).squeeze()
        
        return mic_value

    def generate(self, idx):

        # Generate New Peptide Sequence:
        while True:
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_new = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_new], dim=1)
            if idx_new.item() == 21:
                break

        return idx