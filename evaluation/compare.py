import pandas as pd
import os
from tabulate import tabulate
from datetime import datetime

# 1. Define your model names and their corresponding training FOLDERS
# Note: Point to the folder, NOT the best.pt file directly
results_paths = {
    "YOLOv5n": r"E:\YOLOV8\runs\detect\train(v5)",
    "YOLOv8n": r"E:\YOLOV8\runs\detect\train(v8)",
    "YOLOv9n": r"E:\YOLOV8\runs\detect\train(v9)",
    "YOLOv10n": r"E:\YOLOV8\runs\detect\train(v10)",
    "YOLOv11n": r"E:\YOLOV8\runs\detect\train(v11)",
    "YOLOv12n": r"E:\YOLOV8\runs\detect\train(v12)",
    "YOLOv26s": r"E:\YOLOV8\runs\detect\train5(26s)",
    "YOLOv26saug": r"E:\YOLOV8\runs\detect\ISL_v2\yoloe26s_full-3",
    "YOLOv26n": r"E:\YOLOV8\runs\detect\train4(26n)",
    "SAMYOLO11": r"E:\YOLOV8\runs\sign_language_seg\yolo11_seg_v1-10",
    "RT-DETR": r"E:\YOLOV8\runs\sign_language_rtdetr\finetune_v1",
    "YOLOE": r"E:\ISL\yoloe_full-2",
    "YOLOWORLD": r"E:\YOLOV8\runs\detect\train"
}

def get_model_info(folder_path):
    # Determine the weights file path (checking for .pt and SuperGradients .pth)
    weights_options = [
        os.path.join(folder_path, 'weights', 'best.pt'),
        os.path.join(folder_path, 'ckpt_best.pth'),
        os.path.join(folder_path, 'weights', 'best.pth')
    ]
    
    weights_file = next((f for f in weights_options if os.path.exists(f)), None)
    csv_path = os.path.join(folder_path, 'results.csv')
    args_path = os.path.join(folder_path, 'args.yaml')

    if not os.path.exists(csv_path) or not weights_file:
        return None

    try:
        # 1. Extract Metrics from CSV
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]
        
        precision = last.get('metrics/precision(B)', last.get('metrics/precision', 0))
        recall = last.get('metrics/recall(B)', last.get('metrics/recall', 0))
        map50 = last.get('metrics/mAP50(B)', last.get('metrics/mAP50', 0))
        map50_95 = last.get('metrics/mAP50-95(B)', last.get('metrics/mAP50-95', 0))
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 2. Get Model Size (MB)
        size_mb = os.path.getsize(weights_file) / (1024 * 1024)

        # 3. Calculate Training Time (Hours)
        # Difference between args.yaml (start) and best.pt (end)
        start_time = os.path.getctime(args_path) if os.path.exists(args_path) else None
        end_time = os.path.getmtime(weights_file)
        
        if start_time:
            duration_hrs = (end_time - start_time) / 3600
        else:
            duration_hrs = 0

        return {
            "P": round(precision, 4),
            "R": round(recall, 4),
            "F1": round(f1, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "Size": round(size_mb, 2),
            "Time": round(duration_hrs, 2)
        }
    except Exception as e:
        return None

if __name__ == "__main__":
    final_data = []

    for model_name, path in results_paths.items():
        m = get_model_info(path)
        if m:
            final_data.append([
                model_name, m["Size"], m["Time"], m["P"], m["R"], m["F1"], m["mAP50"], m["mAP50_95"]
            ])
        else:
            final_data.append([model_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

    headers = ["Model", "Size (MB)", "Time (Hr)", "Precision", "Recall", "F1 Score", "mAP50", "mAP50-95"]
    
    print("\n" + "="*110)
    print("                INDIAN SIGN LANGUAGE (ISL) COMPREHENSIVE PERFORMANCE REPORT")
    print("="*110)
    print(tabulate(final_data, headers=headers, tablefmt="grid"))
    print("="*110)

    # Save to CSV
    report_df = pd.DataFrame(final_data, columns=headers)
    report_df.to_csv(r'E:\YOLOV8\ISL_Full_Technical_Report.csv', index=False)
