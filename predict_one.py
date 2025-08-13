#!/usr/bin/env python3
import argparse, torch
from PIL import Image
from torchvision import transforms
from models.efficientnet_b0 import get_efficientnet_b0

def load_ckpt(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        classes  = ckpt.get("classes")
        img_size = ckpt.get("img_size", 224)
        state    = ckpt.get("state_dict") or ckpt.get("model_state") or ckpt.get("model") or ckpt
    else:
        classes, img_size, state = None, 224, ckpt

    num_classes = len(classes) if classes else 10  # 기본값 10
    try:
        model = get_efficientnet_b0(num_classes=num_classes, pretrained=False).to(device)
    except TypeError:
        model = get_efficientnet_b0(num_classes).to(device)

    # prefix 제거
    new = {}
    for k,v in state.items():
        if k.startswith("module."): k = k[7:]
        if k.startswith("model."):  k = k[6:]
        new[k]=v
    model.load_state_dict(new, strict=False)
    model.eval()
    return model, classes, img_size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, classes, img_size = load_ckpt(args.ckpt, args.device)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(args.device)
    with torch.no_grad():
        logits = model(x)
        prob = logits.softmax(1)[0]
        topk = torch.topk(prob, k=min(3, prob.numel()))
    idxs = topk.indices.cpu().tolist()
    probs = topk.values.cpu().tolist()
    names = [classes[i] if classes else str(i) for i in idxs]

    print("\nTop-k:")
    for n, p in zip(names, probs):
        print(f"{n}: {p*100:.2f}%")

if __name__ == "__main__":
    main()
