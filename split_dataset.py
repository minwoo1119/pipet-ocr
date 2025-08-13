#!/usr/bin/env python3
import argparse, random, shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="현재 0..9 폴더가 있는 루트")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val",   type=float, default=0.15)
    ap.add_argument("--test",  type=float, default=0.15)
    ap.add_argument("--seed",  type=int,   default=42)
    ap.add_argument("--move", action="store_true", help="복사 대신 이동")
    args = ap.parse_args()

    if round(args.train + args.val + args.test, 6) != 1.0:
        raise ValueError("train+val+test 비율 합이 1이어야 합니다.")

    root = Path(args.data_dir)
    split_names = {"train", "val", "test"}
    class_dirs = [d for d in root.iterdir() if d.is_dir() and d.name not in split_names]
    if not class_dirs:
        print(f"[WARN] {root} 아래에 클래스 폴더가 없습니다.")
        return

    rng = random.Random(args.seed)

    for cdir in sorted(class_dirs):
        files = [p for p in cdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        rng.shuffle(files)
        n = len(files)
        n_tr = int(n * args.train)
        n_va = int(n * args.val)
        n_te = n - n_tr - n_va

        splits = {
            "train": files[:n_tr],
            "val":   files[n_tr:n_tr+n_va],
            "test":  files[n_tr+n_va:]
        }

        for split, flist in splits.items():
            outdir = root / split / cdir.name
            outdir.mkdir(parents=True, exist_ok=True)
            for src in flist:
                dst = outdir / src.name
                if args.move:
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))

        print(f"{cdir.name}: total={n}, train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    print("\n[OK] 분리 완료 →", root / "train", root / "val", root / "test")
    if not args.move:
        print("[INFO] 원본( data/0..9 )은 그대로 두고, 학습은 data/train 과 data/val 을 사용하세요.")

if __name__ == "__main__":
    main()
