import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans, Samples,
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
                 norm_rate=0.25, depth_norm=1.0):
        super().__init__()
        self.norm_rate = norm_rate
        self.depth_norm = depth_norm

        self.conv1     = nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False)
        self.bn1       = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.bn2       = nn.BatchNorm2d(F1 * D)
        self.pool1     = nn.AvgPool2d((1, 4))
        self.drop1     = nn.Dropout(dropoutRate)

        self.sep_dw    = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same",
                                   groups=F1 * D, bias=False)
        self.sep_pw    = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3       = nn.BatchNorm2d(F2)
        self.pool2     = nn.AvgPool2d((1, 8))
        self.drop2     = nn.Dropout(dropoutRate)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            feat_dim = self._features(dummy).shape[1]
        self.classifier = nn.Linear(feat_dim, nb_classes)

    def _features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.sep_dw(x)
        x = self.sep_pw(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x.flatten(1)

    def forward(self, x):
        return self.classifier(self._features(x))


def load_encoder(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = EEGNet(
        nb_classes=cfg["nb_classes"],
        Chans=cfg["Chans"],
        Samples=cfg["Samples"],
        dropoutRate=cfg.get("dropoutRate", 0.5),
        kernLength=cfg["kernLength"],
        F1=cfg["F1"], D=cfg["D"], F2=cfg["F2"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt