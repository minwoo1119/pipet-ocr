import cv2

# ROI 좌표는 고정값 (x, y, w, h) 형태
ROI_COORDS = [
    (10, 20, 50, 80),
    (10,120, 50, 80),
    (10,220, 50, 80),
    (10,320, 50, 80)
]

def crop_rois(image):
    rois = []
    for (x, y, w, h) in ROI_COORDS:
        roi = image[y:y+h, x:x+w]
        rois.append(roi)
    return rois

if __name__ == "__main__":
    img = cv2.imread("sample_pipette.jpg")
    crops = crop_rois(img)
    for i, c in enumerate(crops):
        cv2.imwrite(f"roi_{i}.jpg", c)
