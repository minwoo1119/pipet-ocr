import timm
import torch.nn as nn

def get_efficientnet_b0(num_classes: int):
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    return model
