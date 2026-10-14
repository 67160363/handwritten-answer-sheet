"""ขั้น 4: เทรน CNN ที่ propose — DigitNet-T3 (11 คลาส: 0-9 + blank)
รัน:  python step4_train_cnn.py [--no-mnist] [--epochs 30]
ผล:  out/digitnet_t3.pt, out/cnn_report.txt, out/cnn_confusion.png, out/cnn_curve.png
"""
import os, argparse
import numpy as np, pandas as pd, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import datasets
from common import ROOT, OUT, CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--no-mnist", action="store_true", help="ไม่ pre-train ด้วย MNIST")
ap.add_argument("--epochs", type=int, default=30)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)
torch.manual_seed(0)


from digitnet import DigitNet, NORM, aug, plain


class CellDS(Dataset):
    def __init__(self, df, tf):
        self.p = [os.path.join(ROOT, p) for p in df.path]
        self.y = [CLASSES.index(str(l)) for l in df.label]
        self.tf = tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return self.tf(cv2.imread(self.p[i], 0)), self.y[i]


def evaluate(model, loader):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            P += model(x.to(dev)).argmax(1).cpu().tolist(); Y += y.tolist()
    return np.array(P), np.array(Y)


def fit(model, tr, va, epochs, lr, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(tr))
    best, hist, path = -1, [], os.path.join(OUT, f"{tag}.pt")
    for ep in range(epochs):
        model.train(); n = ok = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            out = model(x); loss = F.cross_entropy(out, y, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            ok += (out.argmax(1) == y).sum().item(); n += len(y)
        P, Y = evaluate(model, va); va_acc = (P == Y).mean() if len(Y) else 0
        hist.append((ok / n, va_acc))
        if va_acc >= best:
            best = va_acc; torch.save(model.state_dict(), path)
        print(f"[{tag}] ep{ep+1:02d} train {ok/n:.4f} val {va_acc:.4f}")
    model.load_state_dict(torch.load(path, map_location=dev))
    return hist


df = pd.read_csv(os.path.join(OUT, "cells.csv"), dtype={"label": str})
print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
dl = {s: DataLoader(CellDS(df[df.split == s], aug if s == "train" else plain), batch_size=128,
                    shuffle=(s == "train")) for s in ["train", "val", "test"]}

model = DigitNet().to(dev)
print("params:", sum(p.numel() for p in model.parameters()))
if not args.no_mnist:
    mt = T.Compose([T.Pad(2), T.RandomAffine(8, translate=(0.08, 0.08), scale=(0.85, 1.1)), T.ToTensor(), NORM])
    mv = T.Compose([T.Pad(2), T.ToTensor(), NORM])
    mroot = os.path.join(ROOT, "data", "mnist")
    fit(model,
        DataLoader(datasets.MNIST(mroot, True, mt, download=True), 256, shuffle=True),
        DataLoader(datasets.MNIST(mroot, False, mv, download=True), 512),
        epochs=3, lr=3e-3, tag="digitnet_mnist")

hist = fit(model, dl["train"], dl["val"], args.epochs, 2e-3, "digitnet_t3")

# ---- รายงานผลบน test (แผ่นที่ไม่เคยเห็น)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P, Y = evaluate(model, dl["test"])
rep = classification_report(Y, P, labels=range(11), target_names=CLASSES, digits=4, zero_division=0)
print(rep); open(os.path.join(OUT, "cnn_report.txt"), "w", encoding="utf-8").write(rep)
ConfusionMatrixDisplay(confusion_matrix(Y, P, labels=range(11)), display_labels=CLASSES).plot(cmap="Blues")
plt.title("DigitNet-T3 — test"); plt.savefig(os.path.join(OUT, "cnn_confusion.png"), dpi=120); plt.close()
h = np.array(hist)
plt.plot(h[:, 0], label="train"); plt.plot(h[:, 1], label="val"); plt.legend(); plt.grid()
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.savefig(os.path.join(OUT, "cnn_curve.png"), dpi=120)
if (df.split == "test").any() and (df[df.split == "test"].sheet.isin(df[df.split == "train"].sheet)).any():
    print("\n⚠ test มีแผ่นเดียวกับ train (โหมด by_row) — accuracy สูงเกินจริง เก็บข้อมูลเพิ่มก่อนรายงานผล")
