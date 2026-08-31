"""Person detection with YOLO11n running on ONNX Runtime, CPU only.

The model is the stock Ultralytics YOLO11n exported to ONNX. It is a general
80-class COCO detector; we keep class 0 ("person") and discard the rest.

Output layout, for a 320x320 input:

    (1, 84, 2100)
     |   |   +-- one column per anchor point: (320/8)^2 + (320/16)^2 + (320/32)^2
     |   +------ 4 box values (cx, cy, w, h) followed by 80 class scores
     +---------- batch of 1

Two details are worth knowing because they are easy to get wrong:

* the class filter runs **before** NMS, so the expensive overlap comparison
  only ever sees the handful of boxes that were people, not thousands;
* the image is letterboxed (aspect ratio preserved, padded to a square) rather
  than squashed, and the padding has to be undone afterwards. Getting that
  inverse wrong puts boxes slightly off in a way that still looks plausible.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import onnxruntime as ort

PERSON_CLASS = 0


class PersonDetector:
    def __init__(self, model_path: str, conf: float = 0.35, iou: float = 0.5,
                 threads: int = 4):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"])

        spec = self.session.get_inputs()[0]
        self.input_name = spec.name
        self.size = int(spec.shape[2])      # square input, e.g. 320
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.last_ms = 0.0
        # Allocated once and reused: this buffer is filled every single frame.
        self._blob = np.empty((1, 3, self.size, self.size), dtype=np.float32)

    def _letterbox(self, frame):
        """Scale to fit the square input, pad the rest, keep the aspect ratio.

        Returns the model input plus what is needed to map boxes back:
        the scale factor and the padding offset.
        """
        h, w = frame.shape[:2]
        s = self.size
        scale = min(s / w, s / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        pad_x, pad_y = (s - nw) // 2, (s - nh) // 2

        canvas = np.full((s, s, 3), 114, dtype=np.uint8)    # neutral grey padding
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(
            frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, HWC to CHW, 0-255 to 0-1, straight into the reused buffer.
        np.copyto(self._blob[0], canvas[:, :, ::-1].transpose(2, 0, 1), casting="unsafe")
        self._blob /= 255.0
        return self._blob, scale, pad_x, pad_y

    def _decode(self, output):
        """(1, 84, N) -> person boxes in model coordinates, plus scores."""
        pred = output[0]                        # (84, N)
        scores = pred[4 + PERSON_CLASS]         # just the person row
        keep = scores > self.conf
        if not keep.any():
            return np.empty((0, 4), np.float32), np.empty((0,), np.float32)

        cx, cy, w, h = pred[0][keep], pred[1][keep], pred[2][keep], pred[3][keep]
        boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)   # NMS wants xywh
        scores = scores[keep].astype(np.float32)

        keep_idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf, self.iou)
        if len(keep_idx) == 0:
            return np.empty((0, 4), np.float32), np.empty((0,), np.float32)

        b = boxes[np.asarray(keep_idx).reshape(-1)]
        xyxy = np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)
        return xyxy, scores[np.asarray(keep_idx).reshape(-1)]

    def detect(self, frame):
        """Return [(x1, y1, x2, y2, score), ...] in original frame pixels."""
        blob, scale, pad_x, pad_y = self._letterbox(frame)

        t0 = time.perf_counter()
        output = self.session.run(None, {self.input_name: blob})[0]
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        xyxy, scores = self._decode(output)
        if len(xyxy) == 0:
            return []

        # Undo the letterbox: remove the padding, then remove the scaling.
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale

        h, w = frame.shape[:2]
        np.clip(xyxy[:, [0, 2]], 0, w - 1, out=xyxy[:, [0, 2]])
        np.clip(xyxy[:, [1, 3]], 0, h - 1, out=xyxy[:, [1, 3]])

        return [(float(a), float(b), float(c), float(d), float(s))
                for (a, b, c, d), s in zip(xyxy, scores)]
