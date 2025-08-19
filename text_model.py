import torch
import torch.nn as nn

class TextBackbone(nn.Module):
    def __init__(self, vocab_size=30522, emb_dim=128, hidden=128, out_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden*2, out_dim)

    def forward(self, ids, mask=None):
        e = self.emb(ids)
        if mask is not None:
            e = e * mask.unsqueeze(-1).float()
        out, _ = self.rnn(e)
        h = out[:, -1]
        z = self.proj(h)
        return z / (z.norm(dim=1, keepdim=True) + 1e-6)
