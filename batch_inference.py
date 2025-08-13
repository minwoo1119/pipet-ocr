from trt_inference import TRTModel
import numpy as np

trt_model = TRTModel("model_fp16.engine", 224, [0.485,0.456,0.406], [0.229,0.224,0.225])
def predict_batch(images):
    logits = trt_model.infer(images)
    preds = np.argmax(logits, axis=1)
    return preds
