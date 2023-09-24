from ultralytics import YOLO
import cv2
import numpy as np

model_path = 'best.pt'

model = YOLO(model_path)


while True:

    image_path = input('Enter image path: ')

    img = cv2.imread(image_path)
    H, W, _ = img.shape

    results = model(img)

    for result in results:
        if result.masks is not None:
            for j, mask in enumerate(result.masks.data):
                mask = mask.cpu().numpy() * 255  # Move tensor to CPU and then convert to numpy
                mask = cv2.resize(mask, (W, H))
                mask = np.uint8(mask)  # Convert mask to 8-bit unsigned integer
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert grayscale mask to BGR
                overlay = cv2.addWeighted(img, 0.6, mask, 0.4, 0)  # Create overlay
                cv2.imwrite(f'./output_{j}.png', overlay)  # Save overlay image
        else:
            print('No masks found')