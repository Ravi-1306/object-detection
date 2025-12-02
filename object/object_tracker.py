
import cv2
import sys
import argparse
import logging
import numpy as np
import time
from typing import List, Tuple
try:
    import mediapipe as mp
except ImportError:
    mp = None

# Optional: torchvision classifier for general objects
try:
    import torch
    from PIL import Image
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
except Exception:
    torch = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def main():
    parser = argparse.ArgumentParser(description="Hand and face detection with MediaPipe")
    parser.add_argument("--source", default="0", help="Video source: webcam index (e.g., 0) or path to video file")
    parser.add_argument("--display", action="store_true", help="Show display window")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--smooth", type=float, default=0.2, help="EMA smoothing factor for bounding box [0-1]")
    parser.add_argument("--max_hands", type=int, default=2, help="Maximum number of hands to detect")
    parser.add_argument("--classify_interval", type=float, default=0.75, help="Seconds between object classification attempts")
    parser.add_argument("--topk", type=int, default=3, help="Top-K classes to display with percentages")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if mp is None:
        print("Error: mediapipe not installed. Install with 'pip install mediapipe'.")
        sys.exit(1)

    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # Init classifier (ImageNet) if available
    classifier = None
    preprocess = None
    class_labels: List[str] = []
    use_classifier = False
    if torch is not None:
        try:
            weights = MobileNet_V3_Large_Weights.DEFAULT
            classifier = mobilenet_v3_large(weights=weights).eval()
            preprocess = weights.transforms()
            class_labels = weights.meta.get("categories", [])
            use_classifier = True
            logging.info("Loaded MobileNetV3 classifier (%d classes)", len(class_labels) or 1000)
        except Exception as e:
            logging.warning("Classifier init failed: %s", e)

    # Open source
    source = args.source
    if source.isdigit():
        cam_index = int(source)
        logging.info("Opening webcam index %d", cam_index)
        video = cv2.VideoCapture(cam_index)
    else:
        logging.info("Opening video file %s", source)
        video = cv2.VideoCapture(source)

    if not video.isOpened():
        print("Error: Could not open video.")
        sys.exit()

    ema_bboxes = {}  # hand index -> (x,y,w,h)
    ema_face_bbox = None  # (x,y,w,h)
    last_classify_ts = 0.0
    last_topk: List[Tuple[str, float]] = []

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    face_detection = mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    )

    logging.info("Starting detection loop...")
    while True:
        ret, frame = video.read()
        if not ret:
            break

        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_result = hands.process(rgb)
        face_result = face_detection.process(rgb)

        h, w, _ = frame.shape

        # Track faces
        if face_result.detections:
            for detection in face_result.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                
                # Clamp to frame bounds
                x = max(0, x)
                y = max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)
                
                face_bbox = (x, y, bw, bh)
                
                # EMA smoothing for face
                if ema_face_bbox is None:
                    ema_face_bbox = face_bbox
                else:
                    alpha = args.smooth
                    ema_face_bbox = tuple(int(alpha * nb + (1 - alpha) * ob) for nb, ob in zip(face_bbox, ema_face_bbox))
                
                # Draw face bbox
                fx, fy, fw, fh = ema_face_bbox
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
                cv2.putText(frame, "Face", (fx, max(fy - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Track hands
        if hand_result.multi_hand_landmarks is not None:
            for idx, lm in enumerate(hand_result.multi_hand_landmarks[:args.max_hands]):
                xs = [int(p.x * w) for p in lm.landmark]
                ys = [int(p.y * h) for p in lm.landmark]
                x_min, x_max = max(min(xs), 0), min(max(xs), w - 1)
                y_min, y_max = max(min(ys), 0), min(max(ys), h - 1)
                x, y = x_min, y_min
                bw, bh = x_max - x_min, y_max - y_min

                # Add padding
                pad = 20
                x = max(x - pad, 0)
                y = max(y - pad, 0)
                bw = min(bw + 2 * pad, w - x)
                bh = min(bh + 2 * pad, h - y)

                bbox = (x, y, bw, bh)

                # EMA smoothing per hand index
                prev = ema_bboxes.get(idx)
                if prev is None:
                    ema_bboxes[idx] = bbox
                else:
                    alpha = args.smooth
                    ema_bboxes[idx] = tuple(int(alpha * nb + (1 - alpha) * ob) for nb, ob in zip(bbox, prev))

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    frame,
                    lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                # Draw smoothed bbox
                ex, ey, ew, eh = ema_bboxes[idx]
                color = (0, 255, 255) if idx == 0 else (255, 0, 255)  # cyan/magenta
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), color, 2)
                cv2.putText(frame, f"Hand {idx+1}", (ex, max(ey - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Only classify when no hands or faces detected
        has_hands = hand_result.multi_hand_landmarks is not None and len(hand_result.multi_hand_landmarks) > 0
        has_faces = face_result.detections is not None and len(face_result.detections) > 0
        
        if not has_hands and not has_faces:
            cv2.putText(frame, "No hand/face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Fallback: classify scene objects (top-K) at intervals
            if use_classifier and (now - last_classify_ts) >= args.classify_interval:
                last_classify_ts = now
                try:
                    side = min(h, w)
                    y0 = (h - side) // 2
                    x0 = (w - side) // 2
                    crop = frame[y0:y0+side, x0:x0+side]
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_crop)
                    input_tensor = preprocess(img)
                    input_batch = input_tensor.unsqueeze(0)
                    with torch.inference_mode():
                        logits = classifier(input_batch)
                        probs = torch.softmax(logits[0], dim=0)
                        k = max(1, min(args.topk, probs.shape[0]))
                        topk_probs, topk_idxs = torch.topk(probs, k=k)
                        last_topk = []
                        for p, idx_val in zip(topk_probs.tolist(), topk_idxs.tolist()):
                            label = class_labels[idx_val] if class_labels and idx_val < len(class_labels) else str(idx_val)
                            last_topk.append((label, p))
                except Exception as e:
                    logging.debug("Classification error: %s", e)

            # Render last classification results
            if last_topk:
                base_y = 60
                for i, (label, prob) in enumerate(last_topk):
                    txt = f"{label}: {prob*100:.1f}%"
                    cv2.putText(frame, txt, (20, base_y + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if args.display:
            cv2.imshow("Hands / Face / Objects", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    video.release()
    hands.close()
    face_detection.close()
    if args.display:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
