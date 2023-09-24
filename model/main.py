from ultralytics import YOLO
import torch

if __name__ == '__main__':

    print(torch.cuda.is_available())
    # Load a model
    # model = YOLO('yolov8-seg.yaml').load("model5.pt")
    model = YOLO('yolov8-seg.yaml')
    # Train the model
    # results = model.train(data='config.yaml', epochs=100, imgsz=500, batch = 4)
    results = model.train(data="config.yaml", epochs=500, imgsz=600, batch = 8, close_mosaic=0, save_period=20)
