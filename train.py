import os, random, numpy as np, contextlib
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.efficientnet_b0 import get_efficientnet_b0
from models.utils import save_checkpoint

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 1e-4
SEED       = 42

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_loaders(device):
    # 숫자 분류니까 좌우반전은 제외
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomApply([transforms.ColorJitter(0.2,0.2)], p=0.7),
        transforms.RandomAffine(degrees=7, translate=(0.06,0.06), scale=(0.95,1.05)),
        transforms.GaussianBlur(3, sigma=(0.1,1.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02,0.08), ratio=(0.3,3.3)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder("data/train", transform=train_tf)
    val_ds   = datasets.ImageFolder("data/val",   transform=val_tf)

    # Windows 안전: CPU면 num_workers=0 권장
    if os.name == "nt" and device != "cuda":
        nw = 0
    else:
        nw = max(2, (os.cpu_count() or 2) // 2)

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=nw, pin_memory=pin)
    return train_ds, val_ds, train_loader, val_loader

def main():
    seed_everything(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)

    train_ds, val_ds, train_loader, val_loader = build_loaders(DEVICE)
    num_classes = len(train_ds.classes)

    try:
        model = get_efficientnet_b0(num_classes=num_classes, pretrained=True).to(DEVICE)
    except TypeError:
        model = get_efficientnet_b0(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 최신 API로 변경
    amp_ctx = (torch.amp.autocast("cuda") if DEVICE=="cuda" else contextlib.nullcontext())
    scaler = (torch.amp.GradScaler("cuda") if DEVICE=="cuda" else None)

    def train_one_epoch():
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                logits = model(imgs)
                loss = criterion(logits, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += imgs.size(0)
        return loss_sum/total, correct/total

    @torch.no_grad()
    def evaluate():
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * imgs.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += imgs.size(0)
        return loss_sum/total, correct/total

    best_acc, patience, no_imp = 0.0, 5, 0
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch()
        va_loss, va_acc = evaluate()
        scheduler.step()
        print(f"[{epoch:02d}/{EPOCHS}] train {tr_acc:.4f}/{tr_loss:.4f} | val {va_acc:.4f}/{va_loss:.4f}")

        if va_acc > best_acc:
            best_acc, no_imp = va_acc, 0
            save_checkpoint(model, train_ds.classes, IMG_SIZE, "best.pt")
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stopping (patience={patience})")
                break
    print("Best val acc:", best_acc)

if __name__ == "__main__":
    main()
