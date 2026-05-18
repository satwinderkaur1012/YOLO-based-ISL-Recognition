
from ultralytics import YOLO

# This line is crucial for Windows
if __name__ == '__main__':
    # 1. Load the model
    model = YOLO('yolo26s.pt') 
    # 2. Train
    results = model.train(
        data=r'E:\YOLOV8\ISL YOLO\data.yaml', 
        epochs=300, 
        imgsz=640,
        workers=4  # Limits background processes to prevent crashing
    )