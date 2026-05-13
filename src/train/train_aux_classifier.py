from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class IUClsDataset(Dataset):
    def __init__(self, jsonl_path: str) -> None:
        self.rows = [json.loads(x) for x in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if x.strip()]
        self.tf = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(float(row["label_abnormal"]), dtype=torch.float32)
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train auxiliary normal/abnormal classifier")
    parser.add_argument("--train", default="data/iu_train/train.jsonl")
    parser.add_argument("--val", default="data/iu_train/val.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--out", default="output/aux_classifier.pt")
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x).squeeze(1)
            prob = torch.sigmoid(logits)
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(prob.cpu().numpy().tolist())

    y_true = np.array(ys)
    y_prob = np.array(ps)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = IUClsDataset(args.train)
    val_ds = IUClsDataset(args.val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        metrics = evaluate(model, val_loader, device)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(losses)) if losses else 0.0,
                    **metrics,
                },
                ensure_ascii=False,
            )
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
