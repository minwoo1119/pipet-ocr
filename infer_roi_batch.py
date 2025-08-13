import cv2
from roi_cropper import crop_rois
from batch_inference import predict_batch

img = cv2.imread("sample_pipette.jpg")
rois = crop_rois(img)
preds = predict_batch(rois)
print("Predicted digits:", preds)
