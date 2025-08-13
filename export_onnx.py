import torch
from models.efficientnet_b0 import get_efficientnet_b0
from models.utils import load_checkpoint

DEVICE = "cpu"
model, ckpt = load_checkpoint(get_efficientnet_b0(10), "best.pt", DEVICE)
model.eval()

dummy = torch.randn(1,3,ckpt["img_size"],ckpt["img_size"])
torch.onnx.export(model, dummy, "model.onnx",
                  input_names=["images"], output_names=["logits"],
                  opset_version=13)
