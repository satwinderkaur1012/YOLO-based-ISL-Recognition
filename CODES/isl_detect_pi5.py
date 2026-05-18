"""
=============================================================================
  ISL Real-Time Detection -- Raspberry Pi 5
  Model : best26saug.onnx  (26 classes A-Z)
  Output: [1, 300, 6]  [x1, y1, x2, y2, conf, cls]
=============================================================================
  Controls:
    + / -   Raise / lower confidence threshold
    T       Toggle top-3 candidates overlay
    S       Save current frame
    Q / ESC Quit
=============================================================================
"""

import collections
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import onnxruntime as ort

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "best26saug.onnx"
CAMERA_ID   = 0
CAM_W       = 640
CAM_H       = 480
CAM_FPS     = 30
CONF_THRESH = 0.40
IOU_THRESH  = 0.60
MIN_BOX_PX  = 20
MAX_BOX_PX  = 600
SKIP        = 2          # infer every (SKIP+1)th frame
THREADS     = 4
STAB_WINDOW = 5
STAB_THRESH = 0.5

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = len(CLASSES)

_PALETTE = [
    (255, 56,  56),  (255,157,151),  (255,112, 31),  (255,178, 29),
    (207,210, 49),   ( 72,249, 10),  (146,204, 23),  ( 61,219,134),
    ( 26,147, 52),   (  0,212,187),  ( 44,153,168),  (  0,194,255),
    ( 52, 69,147),   (100,115,255),  (  0, 24,236),  (132, 56,255),
    ( 82,  0,133),   (203, 56,255),  (255,149,200),  (255, 55,199),
    (255,255,  0),   (  0,255,255),  (255,  0,255),  (128,128,  0),
    (  0,128,128),   (128,  0,128),
]

def get_color(cls_id):
    return _PALETTE[int(cls_id) % len(_PALETTE)]


# ── Prediction Stabiliser ─────────────────────────────────────────────────────
class PredictionStabiliser:
    def __init__(self, window=5, vote_thresh=0.5, min_conf=0.15):
        self.window      = window
        self.vote_thresh = vote_thresh
        self.min_conf    = min_conf
        self._buffers    = collections.defaultdict(
            lambda: collections.deque(maxlen=window)
        )

    def _grid_key(self, box, grid=5):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2 // grid, (y1 + y2) // 2 // grid)

    def update(self, detections):
        updated = set()
        for box, conf, cls_id in detections:
            if conf < self.min_conf:
                continue
            key = self._grid_key(box)
            self._buffers[key].append((cls_id, conf, box))
            updated.add(key)

        for key in list(self._buffers):
            if key not in updated:
                self._buffers[key].append(None)

        stabilised = []
        for key, buf in self._buffers.items():
            entries = [e for e in buf if e is not None]
            if not entries:
                continue
            votes = collections.Counter(e[0] for e in entries)
            best_cls, best_votes = votes.most_common(1)[0]
            if best_votes / self.window < self.vote_thresh:
                continue
            cls_entries = [e for e in entries if e[0] == best_cls]
            mean_conf   = float(np.mean([e[1] for e in cls_entries]))
            last_box    = cls_entries[-1][2]
            stabilised.append((last_box, mean_conf, best_cls))
        return stabilised

    def reset(self):
        self._buffers.clear()


# ── Camera ────────────────────────────────────────────────────────────────────
def open_camera(cam_id=0):
    import glob
    sources = [cam_id] + sorted(
        glob.glob("/dev/video*"),
        key=lambda p: int(p.replace("/dev/video", ""))
    )
    for src in sources:
        try:
            idx = int(src)
        except (ValueError, TypeError):
            idx = src
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print(f"  Camera: {src}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            return cap
        cap.release()
    print("  ERROR: No working camera found.")
    return None


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads     = THREADS
    opts.inter_op_num_threads     = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(path, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    inp      = sess.get_inputs()[0]
    inp_name = inp.name
    shape    = inp.shape
    imgsz    = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 640
    out_shape = sess.get_outputs()[0].shape
    print(f"  Model    : {Path(path).name}")
    print(f"  Input    : {inp.shape}  ({inp.type})")
    print(f"  Output   : {out_shape}")
    return sess, inp_name, imgsz


# ── Preprocessor ──────────────────────────────────────────────────────────────
class Preprocessor:
    def __init__(self, imgsz=640):
        self.imgsz  = imgsz
        self.canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)

    def __call__(self, frame):
        h, w   = frame.shape[:2]
        scale  = self.imgsz / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        py, px = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        self.canvas[:] = 114
        self.canvas[py:py+nh, px:px+nw] = cv2.resize(frame, (nw, nh))
        rgb    = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(
            rgb.astype(np.float32) / 255.0
        ).transpose(2, 0, 1)[np.newaxis]
        return tensor, scale, px, py


# ── Postprocess ───────────────────────────────────────────────────────────────
def postprocess(raw, orig_h, orig_w, scale, pad_x, pad_y, conf_thresh):
    """
    Handles output shape [1, 300, 6]: [x1, y1, x2, y2, conf, cls_id]
    Coords are in model input space (0-640), letterboxed.
    """
    out = raw[0]   # [300, 6]

    # also handle [1, 6, 300] just in case
    if out.ndim == 2 and out.shape[0] == 6:
        out = out.T

    boxes, scores, class_ids = [], [], []

    for row in out:
        if len(row) < 6:
            continue
        x1_m, y1_m, x2_m, y2_m = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        conf   = float(row[4])
        cls_id = int(round(float(row[5])))

        if conf < conf_thresh:
            continue
        if cls_id < 0 or cls_id >= NUM_CLASSES:
            continue

        # Remove letterbox padding and scale back to original frame
        x1 = int((x1_m - pad_x) / scale)
        y1 = int((y1_m - pad_y) / scale)
        x2 = int((x2_m - pad_x) / scale)
        y2 = int((y2_m - pad_y) / scale)

        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w,     x2))
        y2 = max(0, min(orig_h,     y2))

        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
            continue
        if bw > MAX_BOX_PX or bh > MAX_BOX_PX:
            continue

        boxes.append([x1, y1, bw, bh])
        scores.append(conf)
        class_ids.append(cls_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, IOU_THRESH)
    return [
        ([boxes[i][0], boxes[i][1],
          boxes[i][0] + boxes[i][2],
          boxes[i][1] + boxes[i][3]],
         scores[i], class_ids[i])
        for i in indices
    ]


def get_top3(raw, conf_thresh=0.10):
    out = raw[0]
    if out.ndim == 2 and out.shape[0] == 6:
        out = out.T
    rows = []
    for row in out:
        if len(row) < 6:
            continue
        conf   = float(row[4])
        cls_id = int(round(float(row[5])))
        if conf >= conf_thresh and 0 <= cls_id < NUM_CLASSES:
            rows.append((conf, cls_id))
    rows.sort(key=lambda x: -x[0])
    seen, top3 = set(), []
    for conf, cls_id in rows:
        if cls_id not in seen:
            top3.append((conf, cls_id))
            seen.add(cls_id)
        if len(top3) == 3:
            break
    return top3


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_detections(frame, detections):
    for (x1, y1, x2, y2), conf, cls_id in detections:
        color = get_color(cls_id)
        name  = CLASSES[cls_id] if 0 <= cls_id < NUM_CLASSES else str(cls_id)
        label = f"{name}  {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        by1 = max(y1 - th - 8, 0)
        cv2.rectangle(frame, (x1, by1), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_top3(frame, top3, show):
    if not show or not top3:
        return frame
    x, y0 = 10, frame.shape[0] - 10
    items = [(f"#{r+1} {CLASSES[cid]}  {conf:.2f}", r)
             for r, (conf, cid) in enumerate(top3)]
    for text, rank in reversed(items):
        color = (0, 255, 180) if rank == 0 else \
                (200, 200, 200) if rank == 1 else (120, 120, 120)
        sz    = 0.6 if rank == 0 else 0.5
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sz, 1)
        cv2.rectangle(frame, (x-2, y0-th-4), (x+tw+4, y0+2), (20,20,20), -1)
        cv2.putText(frame, text, (x, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, sz, color, 1, cv2.LINE_AA)
        y0 -= th + 8
    return frame


def draw_hud(frame, fps, lat_ms, det_count, conf_thresh, show_top3):
    lines = [
        (f"Model  : ISL YOLOE-26s",    (100, 200, 255)),
        (f"FPS    : {fps:5.1f}",        (200, 200, 200)),
        (f"Lat    : {lat_ms:5.1f}ms",   (200, 200, 200)),
        (f"Dets   : {det_count}",       (200, 200, 200)),
        (f"Conf   : {conf_thresh:.2f}", (200, 200, 200)),
        ("",                            (0, 0, 0)),
        ("+/-=Conf  T=Top3",           (150, 150, 150)),
        ("S=Save  Q=Quit",              (110, 110, 110)),
    ]
    pad, lh = 8, 20
    bh, bw  = lh * len(lines) + pad * 2, 210
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (bw, bh), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + (i+1)*lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)
    t3 = "ON" if show_top3 else "OFF"
    cv2.putText(frame, f"Top3: {t3}", (pad, pad + len(lines)*lh + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110,110,110), 1, cv2.LINE_AA)
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n  ISL Detection -- Raspberry Pi 5")
    print("  " + "="*44)

    if not Path(MODEL_PATH).exists():
        print(f"  ERROR: {MODEL_PATH} not found")
        return

    sess, inp_name, imgsz = load_model(MODEL_PATH)
    pre       = Preprocessor(imgsz)
    stabiliser = PredictionStabiliser(STAB_WINDOW, STAB_THRESH, CONF_THRESH * 0.5)
    conf_thr  = CONF_THRESH
    show_top3 = True

    cap = open_camera(CAMERA_ID)
    if cap is None:
        return

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera   : {aw}x{ah}")
    print(f"  Conf     : {conf_thr}   IOU: {IOU_THRESH}")
    print(f"  Skip     : {SKIP}  (infer every {SKIP+1} frames)")
    print(f"  Threads  : {THREADS}")
    print(f"\n  Keys: +/-=Conf  T=Top3  S=Save  Q/ESC=Quit\n")

    fps_buf    = collections.deque(maxlen=20)
    lat_ms     = 0.0
    fps        = 0.0
    frame_idx  = 0
    save_count = 0
    detections = []
    top3       = []
    last_raw   = None

    while True:
        cap.grab()
        frame_idx += 1

        # ── Skipped frames: redraw last detections ────────────────────────────
        if frame_idx % (SKIP + 1) != 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
            frame = draw_detections(frame, detections)
            if last_raw is not None:
                top3 = get_top3(last_raw, conf_thr * 0.5)
            frame = draw_top3(frame, top3, show_top3)
            frame = draw_hud(frame, fps, lat_ms, len(detections), conf_thr, show_top3)
            cv2.imshow("ISL Detection  |  Pi 5", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            continue

        ret, frame = cap.retrieve()
        if not ret or frame is None:
            break

        orig_h, orig_w = frame.shape[:2]

        # ── Inference ─────────────────────────────────────────────────────────
        tensor, scale, px, py = pre(frame)
        t0   = time.perf_counter()
        outs = sess.run(None, {inp_name: tensor})
        t1   = time.perf_counter()

        lat_ms   = (t1 - t0) * 1000
        fps_buf.append(1.0 / max(t1 - t0, 1e-9))
        fps      = float(np.mean(fps_buf))
        last_raw = outs[0]

        # ── Decode + stabilise ────────────────────────────────────────────────
        raw_dets   = postprocess(outs[0], orig_h, orig_w,
                                 scale, px, py, conf_thr)
        detections = stabiliser.update(raw_dets)
        top3       = get_top3(outs[0], conf_thr * 0.5)

        # ── Draw ──────────────────────────────────────────────────────────────
        frame = draw_detections(frame, detections)
        frame = draw_top3(frame, top3, show_top3)
        frame = draw_hud(frame, fps, lat_ms, len(detections), conf_thr, show_top3)
        cv2.imshow("ISL Detection  |  Pi 5", frame)

        # Console log every 30 inferred frames
        inferred = frame_idx // (SKIP + 1)
        if inferred > 0 and inferred % 30 == 0:
            names = [CLASSES[d[2]] for d in detections
                     if 0 <= d[2] < NUM_CLASSES]
            print(f"  f={frame_idx:5d}  fps={fps:5.1f}  lat={lat_ms:6.1f}ms"
                  f"  dets={names}  conf={conf_thr:.2f}")

        # ── Keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key in (ord('+'), ord('=')):
            conf_thr = min(0.95, round(conf_thr + 0.05, 2))
            stabiliser.reset()
            print(f"  Conf → {conf_thr:.2f}")
        elif key == ord('-'):
            conf_thr = max(0.05, round(conf_thr - 0.05, 2))
            stabiliser.reset()
            print(f"  Conf → {conf_thr:.2f}")
        elif key in (ord('t'), ord('T')):
            show_top3 = not show_top3
            print(f"  Top-3: {'ON' if show_top3 else 'OFF'}")
        elif key in (ord('s'), ord('S')):
            fname = f"isl_det_{save_count:04d}.jpg"
            cv2.imwrite(fname, frame)
            save_count += 1
            print(f"  Saved → {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Done — {frame_idx} frames processed.\n")


if __name__ == "__main__":
    main()
