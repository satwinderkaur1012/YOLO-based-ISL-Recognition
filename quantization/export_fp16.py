from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

model = YOLO(r'E:\YOLOV8\runs\detect\ISL_v2\yoloe26s_full-3\weights\best.pt')

# First export to ONNX (FP32)
model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    dynamic=False,
    opset=12
)

# Then quantize to INT8 separately
quantize_dynamic(
    model_input=r"E:\YOLOV8\runs\detect\ISL_v2\yoloe26s_full-3\weights\best.onnx",
    model_output=r'E:\YOLOV8\runs\detect\ISL_v2\yoloe26s_full-3\weights\best_int8.onnx',
    weight_type=QuantType.QInt8
)

print("Done! Copy best_int8.onnx to Raspberry Pi 5")
