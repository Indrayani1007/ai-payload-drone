"""
Plastic Pollution Monitor — Dashboard Backend API
Receives data from Raspberry Pi detector, stores it, serves dashboard.
"""

import json
import math
import time
import random
import threading
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# ─── In-memory data store (replace with SQLite for persistence) ────────────
MAX_DETECTIONS = 2000
detection_log = deque(maxlen=MAX_DETECTIONS)
pollution_map = {}        # GPS grid cell → aggregated pollution data
session_stats = {
    "start_time": datetime.utcnow().isoformat(),
    "total_frames": 0,
    "total_detections": 0,
    "hotspots": 0,
    "area_covered_km2": 0.0,
    "max_pollution_level": "Clean"
}
LEVEL_ORDER = ["Clean", "Low", "Medium", "High", "Critical"]

# ─── GPS Grid quantization (for heatmap clustering) ────────────────────────
GRID_PRECISION = 4   # ~11m grid cells at equator

def gps_to_cell(lat: float, lon: float) -> str:
    """Round GPS to grid cell key."""
    return f"{round(lat, GRID_PRECISION)},{round(lon, GRID_PRECISION)}"


def update_pollution_map(gps: dict, detections: list, pollution: dict):
    """Aggregate detection data into grid cells for heatmap."""
    cell = gps_to_cell(gps["lat"], gps["lon"])
    if cell not in pollution_map:
        pollution_map[cell] = {
            "lat": gps["lat"],
            "lon": gps["lon"],
            "total_detections": 0,
            "frames": 0,
            "max_score": 0,
            "density_level": "Clean",
            "class_counts": {},
            "last_seen": None
        }

    entry = pollution_map[cell]
    entry["total_detections"] += len(detections)
    entry["frames"] += 1
    entry["last_seen"] = datetime.utcnow().isoformat()

    score = pollution.get("density_score", 0)
    if score > entry["max_score"]:
        entry["max_score"] = score
        entry["density_level"] = pollution.get("density_level", "Clean")

    for d in detections:
        cls = d["class"]
        entry["class_counts"][cls] = entry["class_counts"].get(cls, 0) + 1

    # Update session stats
    session_stats["total_frames"] += 1
    session_stats["total_detections"] += len(detections)
    hotspots = sum(1 for c in pollution_map.values() if c["max_score"] >= 80)
    session_stats["hotspots"] = hotspots

    current_max = session_stats["max_pollution_level"]
    if LEVEL_ORDER.index(pollution.get("density_level", "Clean")) > LEVEL_ORDER.index(current_max):
        session_stats["max_pollution_level"] = pollution.get("density_level", "Clean")


# ─── API Endpoints ──────────────────────────────────────────────────────────

@app.route("/api/detection", methods=["POST"])
def receive_detection():
    """Receive detection data from Raspberry Pi."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    data["received_at"] = datetime.utcnow().isoformat()
    detection_log.append(data)

    gps = data.get("gps", {})
    detections = data.get("detections", [])
    pollution = data.get("pollution", {})

    if gps.get("lat") and gps.get("lon"):
        update_pollution_map(gps, detections, pollution)

    return jsonify({"status": "ok", "stored": len(detection_log)}), 200


@app.route("/api/recent", methods=["GET"])
def get_recent():
    """Last N detection events."""
    n = min(int(request.args.get("n", 50)), 500)
    recent = list(detection_log)[-n:]
    return jsonify({"count": len(recent), "data": recent})


@app.route("/api/heatmap", methods=["GET"])
def get_heatmap():
    """All pollution map cells for heatmap rendering."""
    cells = list(pollution_map.values())
    return jsonify({"count": len(cells), "cells": cells})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Session-level statistics."""
    # Class breakdown
    class_totals = {}
    for cell in pollution_map.values():
        for cls, cnt in cell["class_counts"].items():
            class_totals[cls] = class_totals.get(cls, 0) + cnt

    # Pollution level distribution
    level_dist = {l: 0 for l in LEVEL_ORDER}
    for cell in pollution_map.values():
        level_dist[cell["density_level"]] = level_dist.get(cell["density_level"], 0) + 1

    return jsonify({
        **session_stats,
        "map_cells": len(pollution_map),
        "class_breakdown": class_totals,
        "level_distribution": level_dist,
    })


@app.route("/api/live", methods=["GET"])
def get_live():
    """Most recent detection for live view."""
    if not detection_log:
        return jsonify({"status": "no_data"})
    latest = list(detection_log)[-1]
    return jsonify({"status": "ok", "data": latest})


@app.route("/api/export", methods=["GET"])
def export_data():
    """Export all detection log as JSON."""
    all_data = list(detection_log)
    return jsonify({
        "exported_at": datetime.utcnow().isoformat(),
        "total_records": len(all_data),
        "data": all_data
    })


# ─── Demo Data Injector (for dashboard testing without RPi) ────────────────
class DemoInjector:
    """Inject realistic simulated data for dashboard demo/testing."""

    CLASSES = ["plastic_bottle", "plastic_bag", "plastic_packaging", "foam", "other_plastic"]
    POLLUTION_HOTSPOTS = [
        (19.1985, 73.1930),   # Hotspot 1 (high pollution)
        (19.2010, 73.1955),   # Hotspot 2 (medium pollution)
    ]

    def __init__(self):
        self.running = False
        self.lat = 19.1960
        self.lon = 73.1880
        self.path_idx = 0
        self.waypoints = [
            (19.1960, 73.1880), (19.1975, 73.1900), (19.1990, 73.1920),
            (19.2000, 73.1935), (19.2005, 73.1945), (19.2015, 73.1958),
            (19.2020, 73.1965), (19.2010, 73.1975), (19.1998, 73.1988),
            (19.1985, 73.1995), (19.1975, 73.1980), (19.1965, 73.1965),
        ]

    def _pollution_at(self, lat, lon):
        """Higher pollution near hotspots."""
        for hlat, hlon in self.POLLUTION_HOTSPOTS:
            dist = math.sqrt((lat - hlat)**2 + (lon - hlon)**2)
            if dist < 0.003:
                return "high"
            elif dist < 0.006:
                return "medium"
        return random.choice(["clean", "clean", "clean", "low"])

    def generate_frame(self):
        # Move along path
        target = self.waypoints[self.path_idx % len(self.waypoints)]
        self.lat += (target[0] - self.lat) * 0.3 + random.gauss(0, 0.00003)
        self.lon += (target[1] - self.lon) * 0.3 + random.gauss(0, 0.00003)
        if abs(self.lat - target[0]) < 0.0001 and abs(self.lon - target[1]) < 0.0001:
            self.path_idx += 1

        zone = self._pollution_at(self.lat, self.lon)

        # Determine detections based on zone
        count_map = {"clean": 0, "low": random.randint(1, 3),
                     "medium": random.randint(3, 7), "high": random.randint(7, 15)}
        n = count_map[zone]

        detections = []
        for _ in range(n):
            cls = random.choice(self.CLASSES)
            x1, y1 = random.randint(10, 580), random.randint(10, 440)
            w, h = random.randint(20, 120), random.randint(20, 80)
            detections.append({
                "class": cls,
                "confidence": round(random.uniform(0.45, 0.97), 3),
                "bbox": [x1, y1, x1 + w, y1 + h]
            })

        level_map = {"clean": "Clean", "low": "Low", "medium": "Medium", "high": "High"}
        level = level_map[zone]
        score_map = {"Clean": 0, "Low": 25, "Medium": 55, "High": 80}

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "gps": {
                "lat": round(self.lat, 7),
                "lon": round(self.lon, 7),
                "alt": round(30 + random.gauss(0, 1), 1),
                "fix": True
            },
            "detections": detections,
            "pollution": {
                "count": n,
                "coverage_m2": round(n * random.uniform(0.05, 0.3), 2),
                "density_level": level,
                "density_score": score_map.get(level, 0) + random.randint(-5, 10),
                "items_per_100m2": round(n * 2.5, 1),
                "frame_area_m2": round(random.uniform(45, 55), 1)
            }
        }

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        print("[Demo] Data injector started (interval: 2s)")

    def _loop(self):
        while self.running:
            payload = self.generate_frame()
            detection_log.append(payload)
            gps = payload["gps"]
            update_pollution_map(gps, payload["detections"], payload["pollution"])
            time.sleep(2)


# ─── Serve Frontend ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("../frontend", path)


# ─── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Inject demo data (no RPi needed)")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.demo:
        demo = DemoInjector()
        # Pre-populate with 60 seconds of history
        print("[Demo] Pre-populating 60 seconds of data...")
        for _ in range(30):
            payload = demo.generate_frame()
            detection_log.append(payload)
            update_pollution_map(payload["gps"], payload["detections"], payload["pollution"])
        demo.start()

    print(f"\n[✓] Dashboard backend running at: http://localhost:{args.port}")
    print(f"    Open your browser to: http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
