import cv2
import numpy as np
import onnxruntime as ort
import time

# ── Config ─────────────────────────────────────────────
MODEL_PATH  = '/home/pi/isl/best_int8.onnx'
CONF_THRESH = 0.45
IOU_THRESH  = 0.45
IMGSZ       = 640
CLASSES     = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
# ───────────────────────────────────────────────────────

sess = ort.InferenceSession(
    MODEL_PATH,
    providers=['CPUExecutionProvider']
)
inp_name    = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
print(f"Model loaded | Input: {input_shape}")

def preprocess(frame):
    img = cv2.resize(frame, (IMGSZ, IMGSZ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def xywh2xyxy(boxes, orig_w, orig_h):
    """Convert YOLO xywh (normalized) to pixel xyxy"""
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * orig_w
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * orig_h
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * orig_w
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * orig_h
    return np.stack([x1, y1, x2, y2], axis=1)

def nms(boxes, scores, iou_thresh):
    """Simple NMS"""
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < iou_thresh]
    return keep

def postprocess(output, orig_w, orig_h):
    """Parse YOLO output: shape [1, 30, 8400]"""
    preds = output[0][0].T              # → [8400, 30]
    boxes_raw   = preds[:, :4]          # cx, cy, w, h (normalized)
    class_scores = preds[:, 4:]         # [8400, 26]

    class_ids    = np.argmax(class_scores, axis=1)
    confidences  = class_scores[np.arange(len(class_scores)), class_ids]

    mask = confidences > CONF_THRESH
    if not mask.any():
        return []

    boxes_raw   = boxes_raw[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    # Convert to pixel coords
    boxes_px = xywh2xyxy(boxes_raw, orig_w, orig_h)
    boxes_px = np.clip(boxes_px, 0, [orig_w, orig_h, orig_w, orig_h])

    # NMS
    keep = nms(boxes_px, confidences, IOU_THRESH)

    detections = []
    for i in keep:
        detections.append({
            'box':   boxes_px[i].astype(int),
            'conf':  float(confidences[i]),
            'label': CLASSES[class_ids[i]]
        })
    return detections

def draw(frame, detections, fps):
    for d in detections:
        x1, y1, x2, y2 = d['box']
        label = f"{d['label']} {d['conf']:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f'FPS: {fps:.1f}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    if detections:
        cv2.putText(frame, detections[0]['label'], (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 3)
    return frame

# ── Main loop ──────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Starting... Press Q to quit")
fps      = 0
prev     = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]
    inp  = preprocess(frame)
    out  = sess.run(None, {inp_name: inp})
    dets = postprocess(out, orig_w, orig_h)

    now  = time.time()
    fps  = 0.9 * fps + 0.1 * (1.0 / (now - prev))  # smoothed FPS
    prev = now

    frame = draw(frame, dets, fps)
    cv2.imshow('ISL Detection - Pi5', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
