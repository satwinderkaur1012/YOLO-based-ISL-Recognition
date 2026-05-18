from ultralytics import YOLO

model = YOLO('yolo26s.pt')

model.train(
    data=r'E:\YOLOV8\ISL YOLO\data.yaml',
    epochs=300,
    imgsz=640,
    batch=16,
    patience=30,
    cos_lr=True,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    mosaic=1.0,
    flipud=0.3,
    degrees=15.0,
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.4,
    scale=0.5,
    shear=5.0,
    translate=0.1,
    workers=0,
    pretrained=True,
    project='ISL_v2',
    name='yoloe26s_full'
)
