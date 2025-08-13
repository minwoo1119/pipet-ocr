import torch

def save_checkpoint(model, classes, img_size, path):
    torch.save({
        "model": model.state_dict(),
        "classes": classes,
        "img_size": img_size,
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225]
    }, path)

def load_checkpoint(model, path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt
