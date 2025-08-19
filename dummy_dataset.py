import torch
from torch.utils.data import Dataset
import random

EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','neutral']

class DummyMultimodalDataset(Dataset):
    def __init__(self, num_samples=500, seq_len=16, num_classes=7):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        vision = torch.randn(3, 224, 224)
        audio = torch.randn(1, 64, 64)
        text_ids = torch.randint(1, 10000, (self.seq_len,))
        text_mask = torch.ones(self.seq_len)
        label = random.randint(0, self.num_classes-1)
        return vision, audio, text_ids, text_mask, label

def collate_fn(batch):
    visions, audios, ids, masks, labels = zip(*batch)
    visions = torch.stack(visions)
    audios = torch.stack(audios)
    max_len = max(x.size(0) for x in ids)
    ids_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    masks_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (t, m) in enumerate(zip(ids, masks)):
        ids_pad[i, :t.size(0)] = t
        masks_pad[i, :m.size(0)] = m
    labels = torch.tensor(labels, dtype=torch.long)
    return visions, audios, ids_pad, masks_pad, labels
