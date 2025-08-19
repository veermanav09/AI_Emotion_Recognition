import torch, torch.nn as nn
from torch.utils.data import DataLoader
from models.vision_model import VisionBackbone
from models.audio_model import AudioBackbone
from models.text_model import TextBackbone
from models.fusion_model import FusionClassifier
from data.dummy_dataset import DummyMultimodalDataset, collate_fn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    train_ds = DummyMultimodalDataset(num_samples=256)
    val_ds = DummyMultimodalDataset(num_samples=64)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    vnet, anet, tnet = VisionBackbone().to(DEVICE), AudioBackbone().to(DEVICE), TextBackbone().to(DEVICE)
    clf = FusionClassifier().to(DEVICE)

    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in [vnet, anet, tnet, clf]]
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        vnet.train(); anet.train(); tnet.train(); clf.train()
        for vision, audio, ids, masks, labels in train_loader:
            vision, audio, ids, masks, labels = vision.to(DEVICE), audio.to(DEVICE), ids.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
            for opt in optimizers: opt.zero_grad()
            logits = clf(vnet(vision), anet(audio), tnet(ids, masks))
            loss = criterion(logits, labels)
            loss.backward()
            for opt in optimizers: opt.step()
        print(f"Epoch {epoch+1} finished")

    torch.save({'vnet': vnet.state_dict(),'anet': anet.state_dict(),'tnet': tnet.state_dict(),'clf': clf.state_dict()}, 'checkpoint.pth')
    print("Checkpoint saved to checkpoint.pth")

if __name__ == "__main__":
    main()
