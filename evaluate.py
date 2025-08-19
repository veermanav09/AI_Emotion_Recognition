import torch
from torch.utils.data import DataLoader
from models.vision_model import VisionBackbone
from models.audio_model import AudioBackbone
from models.text_model import TextBackbone
from models.fusion_model import FusionClassifier
from data.dummy_dataset import DummyMultimodalDataset, collate_fn
from utils.evaluation import compute_metrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    ds = DummyMultimodalDataset(num_samples=64)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    vnet, anet, tnet, clf = VisionBackbone().to(DEVICE), AudioBackbone().to(DEVICE), TextBackbone().to(DEVICE), FusionClassifier().to(DEVICE)

    try:
        ckpt = torch.load('checkpoint.pth', map_location=DEVICE)
        vnet.load_state_dict(ckpt['vnet']); anet.load_state_dict(ckpt['anet']); tnet.load_state_dict(ckpt['tnet']); clf.load_state_dict(ckpt['clf'])
        print("Loaded checkpoint.pth")
    except FileNotFoundError:
        print("No checkpoint found. Evaluating randomly-initialized models.")

    vnet.eval(); anet.eval(); tnet.eval(); clf.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for v, a, ids, masks, y in dl:
            v, a, ids, masks = v.to(DEVICE), a.to(DEVICE), ids.to(DEVICE), masks.to(DEVICE)
            logits = clf(vnet(v), anet(a), tnet(ids, masks))
            all_pred.extend(logits.argmax(1).cpu().tolist())
            all_true.extend(y.tolist())

    print(compute_metrics(all_true, all_pred))

if __name__ == "__main__":
    main()
