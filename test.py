import os, torch, argparse, numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from models.efficientnet_b0 import get_efficientnet_b0
from models.utils import load_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--data", default="data/test")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dummy = get_efficientnet_b0(10)
    model, ckpt = load_checkpoint(dummy, args.ckpt, DEVICE)
    model.to(DEVICE).eval()

    img_size = ckpt.get("img_size", 224)
    mean = ckpt.get("norm_mean", [0.485, 0.456, 0.406])
    std  = ckpt.get("norm_std",  [0.229, 0.224, 0.225])
    ckpt_classes = ckpt.get("classes", None)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds = datasets.ImageFolder(args.data, transform=tf)
    data_classes = ds.classes

    if ckpt_classes is not None and data_classes != ckpt_classes:
        print("[WARN] 체크포인트의 클래스 순서와 테스트 폴더 클래스가 다릅니다.")
        print("ckpt :", ckpt_classes)
        print("test :", data_classes)
        print("라벨 매칭이 어긋나면 결과가 틀어질 수 있어요.\n")

    if os.name == "nt" and DEVICE != "cuda":
        num_workers, pin_memory = 0, False
    else:
        num_workers = max(2, (os.cpu_count() or 2)//2)
        pin_memory  = (DEVICE == "cuda")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            logits = model(imgs)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.append(preds)
            y_true.append(labels.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro")
    print(f"\nAccuracy: {acc:.4f} | Macro-F1: {f1m:.4f}\n")

    print("Per-class report:")
    print(classification_report(y_true, y_pred, target_names=[str(c) for c in data_classes], digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (rows=true, cols=pred):\n", cm)

if __name__ == "__main__":
    main()
