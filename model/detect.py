"""
Plastic Pollution Detection Engine - Raspberry Pi 4B Deployment
Runs real-time inference on drone camera feed + GPS tagging + sends data to dashboard
"""

import cv2
import json
import time
import math
import argparse
import threading
import requests
import numpy as np
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────
# INSTALL ON RASPBERRY PI:
# pip install ultralytics opencv-python-headless requests gpsd-py3
# ─────────────────────────────────────────

# ── Configuration ──────────────────────────────────────────────
CONF_THRESHOLD = 0.40       # Minimum confidence to count as detection
IOU_THRESHOLD = 0.45        # NMS IoU threshold
FRAME_SKIP = 2              # Process every Nth frame (performance)
IMGSZ = 416                 # Inference image size
API_URL = "http://localhost:5000/api/detection"  # Dashboard backend
STREAM_PORT = 8080           # MJPEG stream port

CLASSES = [
    "plastic_bottle",
    "plastic_bag",
    "plastic_packaging",
    "foam",
    "other_plastic"
]

CLASS_COLORS = {
    "plastic_bottle":   (0, 80, 255),    # Red-orange
    "plastic_bag":      (0, 165, 255),   # Orange
    "plastic_packaging":(0, 220, 200),   # Yellow
    "foam":             (255, 100, 0),   # Blue
    "other_plastic":    (180, 0, 180),   # Purple
}

# ── GPS Reader ─────────────────────────────────────────────────
class GPSReader:
    """Read GPS coordinates from flight controller via GPSD or serial."""

    def __init__(self, use_gpsd: bool = True, mock: bool = False):
        self.lat = 19.0760     # Default: Mumbai area (change to your field location)
        self.lon = 72.8777
        self.alt = 30.0        # meters
        self.heading = 0.0
        self.fix = False
        self.mock = mock
        self._running = False

        if not mock and use_gpsd:
            self._start_gpsd()
        elif mock:
            self._start_mock()

    def _start_gpsd(self):
        """Connect to GPSD daemon."""
        try:
            import gpsd
            gpsd.connect()
            self._running = True
            t = threading.Thread(target=self._gpsd_loop, daemon=True)
            t.start()
            print("[GPS] Connected to GPSD")
        except Exception as e:
            print(f"[GPS] GPSD failed: {e}. Using mock GPS.")
            self._start_mock()

    def _gpsd_loop(self):
        import gpsd
        while self._running:
            try:
                packet = gpsd.get_current()
                if packet.mode >= 2:
                    self.lat = packet.lat
                    self.lon = packet.lon
                    self.alt = packet.alt if packet.mode == 3 else self.alt
                    self.fix = True
            except Exception:
                pass
            time.sleep(0.5)

    def _start_mock(self):
        """Simulate GPS movement along a river path for demo."""
        self.mock = True
        self._running = True
        t = threading.Thread(target=self._mock_loop, daemon=True)
        t.start()
        print("[GPS] Using simulated GPS path")

    def _mock_loop(self):
        """Simulate drone flying along a river."""
        # Simulate a path along Ulhas River, Thane
        waypoints = [
            (19.1960, 73.1880),
            (19.1975, 73.1900),
            (19.1990, 73.1920),
            (19.2005, 73.1945),
            (19.2020, 73.1960),
            (19.2010, 73.1975),
            (19.1995, 73.1990),
        ]
        i = 0
        while self._running:
            self.lat, self.lon = waypoints[i % len(waypoints)]
            # Add small jitter to simulate real GPS
            self.lat += np.random.normal(0, 0.00005)
            self.lon += np.random.normal(0, 0.00005)
            self.alt = 30 + np.random.normal(0, 1)
            self.fix = True
            i += 1
            time.sleep(3)

    def get(self):
        return {
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "alt": round(self.alt, 1),
            "fix": self.fix,
            "timestamp": datetime.utcnow().isoformat()
        }

    def stop(self):
        self._running = False


# ── Pollution Estimator ─────────────────────────────────────────
class PollutionEstimator:
    """Estimate pollution density from detection bounding boxes."""

    def __init__(self, frame_width: int, frame_height: int, altitude: float = 30.0):
        self.fw = frame_width
        self.fh = frame_height
        # Ground coverage at 30m altitude with typical drone camera FOV (~94°)
        # GSD (Ground Sampling Distance) ≈ altitude * tan(FOV/2) * 2 / image_width
        self.fov_h = math.radians(94)
        self.update_altitude(altitude)

    def update_altitude(self, altitude: float):
        self.altitude = altitude
        # Meters per pixel at this altitude
        self.gsd = (2 * altitude * math.tan(self.fov_h / 2)) / self.fw

    def estimate(self, detections: list) -> dict:
        """Calculate pollution metrics from detections."""
        if not detections:
            return {
                "count": 0,
                "coverage_m2": 0.0,
                "density_level": "Clean",
                "density_score": 0,
                "items_per_100m2": 0.0
            }

        total_pixel_area = sum(
            (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
            for d in detections
        )
        total_real_area = total_pixel_area * (self.gsd ** 2)
        frame_real_area = self.fw * self.fh * (self.gsd ** 2)
        coverage_pct = (total_real_area / frame_real_area) * 100

        count = len(detections)
        items_per_100m2 = (count / frame_real_area) * 100 if frame_real_area > 0 else 0

        # Density scoring
        if count == 0:
            level, score = "Clean", 0
        elif count <= 3 or coverage_pct < 1:
            level, score = "Low", 25
        elif count <= 8 or coverage_pct < 5:
            level, score = "Medium", 55
        elif count <= 15 or coverage_pct < 15:
            level, score = "High", 80
        else:
            level, score = "Critical", 100

        return {
            "count": count,
            "coverage_m2": round(total_real_area, 2),
            "coverage_pct": round(coverage_pct, 3),
            "density_level": level,
            "density_score": score,
            "items_per_100m2": round(items_per_100m2, 2),
            "frame_area_m2": round(frame_real_area, 2)
        }


# ── Detection Engine ────────────────────────────────────────────
class PlasticDetector:
    """Main detection class: model + inference + visualization."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = self._load_model(model_path)
        self.frame_count = 0
        self.fps = 0
        self._fps_time = time.time()

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            print(f"[✓] Model loaded: {model_path}")
            return model
        except ImportError:
            print("[!] Ultralytics not installed. Run: pip install ultralytics")
            return None
        except Exception as e:
            print(f"[!] Model load failed: {e}")
            return None

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """Run inference on frame. Returns annotated frame + detections list."""
        if self.model is None:
            return frame, []

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now = time.time()
            self.fps = 30 / (now - self._fps_time)
            self._fps_time = now

        results = self.model(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMGSZ,
            device=self.device,
            verbose=False
        )

        detections = []
        annotated = frame.copy()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else "unknown"
                color = CLASS_COLORS.get(class_name, (0, 255, 0))

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label background
                label = f"{class_name.replace('_', ' ')} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })

        # Overlay info panel
        self._draw_overlay(annotated, detections)
        return annotated, detections

    def _draw_overlay(self, frame: np.ndarray, detections: list):
        h, w = frame.shape[:2]

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 38), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # FPS and detection count
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
        cv2.putText(frame, f"Detections: {len(detections)}", (120, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (w - 90, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


# ── Data Sender ─────────────────────────────────────────────────
class DataSender:
    """Send detection data to dashboard backend."""

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.log_file = Path("detections_log.jsonl")
        self._queue = []
        self._lock = threading.Lock()
        t = threading.Thread(target=self._send_loop, daemon=True)
        t.start()

    def push(self, gps: dict, detections: list, pollution: dict):
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "gps": gps,
            "detections": detections,
            "pollution": pollution
        }
        with self._lock:
            self._queue.append(payload)
        # Always write to local log
        with open(self.log_file, "a") as f:
            f.write(json.dumps(payload) + "\n")

    def _send_loop(self):
        while True:
            with self._lock:
                if self._queue:
                    payload = self._queue.pop(0)
                else:
                    payload = None
            if payload:
                try:
                    requests.post(self.api_url, json=payload, timeout=2)
                except Exception:
                    pass  # Dashboard not running, data still saved locally
            time.sleep(0.1)


# ── Main Detection Loop ─────────────────────────────────────────
def run_detection(
    model_path: str,
    source: str = "0",
    mock_gps: bool = True,
    send_api: bool = True,
    display: bool = True
):
    print(f"\n{'='*50}")
    print("  Plastic Pollution Detection System")
    print("  Drone AI Payload — Raspberry Pi 4B")
    print(f"{'='*50}\n")

    # Initialize components
    detector = PlasticDetector(model_path)
    gps = GPSReader(mock=mock_gps)
    sender = DataSender(API_URL) if send_api else None

    # Camera source: '0' = USB cam, 'picamera' = Pi camera, or RTSP URL
    if source == "picamera":
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            picam.configure(picam.create_video_configuration(
                main={"size": (640, 480), "format": "BGR888"}
            ))
            picam.start()
            use_picamera = True
            print("[✓] PiCamera2 started")
        except Exception as e:
            print(f"[!] PiCamera2 failed: {e}. Falling back to OpenCV.")
            use_picamera = False
            cap = cv2.VideoCapture(0)
    else:
        use_picamera = False
        src = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    estimator = None
    frame_idx = 0
    last_send = time.time()
    print("[→] Detection running. Press 'q' to quit.\n")

    try:
        while True:
            # Capture frame
            if use_picamera:
                frame = picam.capture_array()
                ret = True
            else:
                ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Initialize estimator on first frame
            if estimator is None:
                h, w = frame.shape[:2]
                gps_data = gps.get()
                estimator = PollutionEstimator(w, h, gps_data["alt"])

            # Skip frames for performance (process every FRAME_SKIP frames)
            if frame_idx % FRAME_SKIP != 0:
                continue

            # Get GPS
            gps_data = gps.get()
            estimator.update_altitude(gps_data["alt"])

            # Run detection
            annotated, detections = detector.detect(frame)

            # Estimate pollution
            pollution = estimator.estimate(detections)

            # Color-code pollution level on frame
            level_color = {
                "Clean": (0, 255, 0),
                "Low": (0, 220, 255),
                "Medium": (0, 165, 255),
                "High": (0, 60, 255),
                "Critical": (0, 0, 200)
            }.get(pollution["density_level"], (255, 255, 255))

            h, w = annotated.shape[:2]
            cv2.putText(annotated,
                        f"GPS: {gps_data['lat']:.5f}, {gps_data['lon']:.5f}",
                        (8, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            cv2.putText(annotated,
                        f"Level: {pollution['density_level']} | Alt: {gps_data['alt']:.0f}m",
                        (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, level_color, 1)

            # Send data every 2 seconds
            if sender and time.time() - last_send >= 2:
                sender.push(gps_data, detections, pollution)
                last_send = time.time()

                if detections:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"GPS({gps_data['lat']:.5f},{gps_data['lon']:.5f}) "
                          f"| {len(detections)} items | Level: {pollution['density_level']}")

            # Display
            if display:
                cv2.imshow("Plastic Pollution Detector", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[→] Stopping detection...")
    finally:
        gps.stop()
        if not use_picamera:
            cap.release()
        cv2.destroyAllWindows()
        print(f"[✓] Detection ended. Log saved to: detections_log.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plastic Pollution Detector - Raspberry Pi")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLOv8 model")
    parser.add_argument("--source", type=str, default="0", help="Camera source: 0, 'picamera', or RTSP URL")
    parser.add_argument("--mock-gps", action="store_true", help="Use simulated GPS")
    parser.add_argument("--no-display", action="store_true", help="Headless mode (for RPi without screen)")
    parser.add_argument("--no-api", action="store_true", help="Disable sending to dashboard API")
    args = parser.parse_args()

    run_detection(
        model_path=args.model,
        source=args.source,
        mock_gps=args.mock_gps,
        send_api=not args.no_api,
        display=not args.no_display
    )
