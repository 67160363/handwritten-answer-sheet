"""DigitNet-T3: CNN ที่ propose สำหรับอ่านตัวเลขลายมือรายช่อง (0-9 + blank)"""
import torch.nn as nn
import torchvision.transforms as T


class DigitNet(nn.Module):
    """3 บล็อก (Conv-BN-ReLU x2 + MaxPool) -> GAP -> Dropout -> FC ; input 1x32x32"""
    def __init__(self, n_cls=11):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_cls))

    def forward(self, x):
        return self.head(self.features(x))


NORM = T.Normalize([0.13], [0.3])
aug = T.Compose([T.ToPILImage(), T.RandomAffine(8, translate=(0.08, 0.08), scale=(0.85, 1.1), shear=8),
                 T.ToTensor(), NORM])
plain = T.Compose([T.ToPILImage(), T.ToTensor(), NORM])
