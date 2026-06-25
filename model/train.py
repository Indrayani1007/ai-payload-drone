"""
Plastic Pollution Detection - YOLOv8 Training Script
Project: AI-Based Plastic Pollution Monitoring for Rivers and Lakes Using a Drone
Hardware: Raspberry Pi 4B + Drone Camera
"""

import os
import yaml
import argparse
from pathlib import Path

# ─────────────────────────────────────────────
# REQUIREMENTS: pip install ultralytics roboflow
# ─────────────────────────────────────────────

def create_dataset_yaml(dataset_path: str) -> str:
    """Create YAML config for YOLOv8 training dataset."""
    yaml_content = {
        "path": dataset_path,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 5,  # number of classes
        "names": [
            "plastic_bottle",
            "plastic_bag",
            "plastic_packaging",
            "foam",
            "other_plastic"
        ]
    }
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"[✓] Dataset YAML created at: {yaml_path}")
    return yaml_path


def download_dataset_roboflow(api_key: str, workspace: str, project: str, version: int, save_dir: str):
    """
    Download annotated plastic waste dataset from Roboflow.
    Replace with your own Roboflow project credentials.
    Public dataset suggestion: search 'plastic waste water' on Roboflow Universe.
    """
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download("yolov8", location=save_dir)
        print(f"[✓] Dataset downloaded to: {save_dir}")
        return dataset.location
    except ImportError:
        print("[!] Roboflow not installed. Run: pip install roboflow")
        return None
    except Exception as e:
        print(f"[!] Dataset download failed: {e}")
        print("    Continuing with manually placed dataset...")
        return save_dir


def train_yolov8(
    yaml_path: str,
    model_size: str = "yolov8n",   # 'n' = nano (lightest, best for RPi)
    epochs: int = 100,
    imgsz: int = 416,               # Reduced for Raspberry Pi inference speed
    batch: int = 16,
    device: str = "0",              # '0' for GPU, 'cpu' for CPU-only
    project_name: str = "plastic_pollution",
    run_name: str = "drone_v1"
):
    """Train YOLOv8 Nano model for floating plastic detection."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] Ultralytics not installed. Run: pip install ultralytics")
        return None

    print(f"\n{'='*50}")
    print(f"  Training YOLOv8 Nano - Plastic Waste Detection")
    print(f"{'='*50}")
    print(f"  Model      : {model_size}.pt")
    print(f"  Epochs     : {epochs}")
    print(f"  Image Size : {imgsz}x{imgsz}")
    print(f"  Device     : {device}")
    print(f"{'='*50}\n")

    # Load pretrained YOLOv8 Nano (downloads automatically)
    model = YOLO(f"{model_size}.pt")

    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_name,
        name=run_name,
        patience=20,             # Early stopping
        save=True,
        save_period=10,
        cache=True,
        augment=True,            # Data augmentation (important for water reflections)
        degrees=15.0,            # Rotation augmentation
        flipud=0.3,              # Vertical flip
        fliplr=0.5,              # Horizontal flip
        mosaic=1.0,              # Mosaic augmentation
        hsv_h=0.015,             # Hue augmentation
        hsv_s=0.7,               # Saturation augmentation (helps with water glare)
        hsv_v=0.4,               # Value augmentation (lighting variations)
        verbose=True,
    )

    # Export to ONNX for Raspberry Pi optimized inference
    best_model_path = f"{project_name}/{run_name}/weights/best.pt"
    if os.path.exists(best_model_path):
        export_model(best_model_path)

    return results


def export_model(model_path: str):
    """Export trained model to ONNX format for RPi deployment."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)

        # Export to ONNX (optimized for edge inference)
        model.export(format="onnx", imgsz=416, simplify=True)
        print(f"\n[✓] Model exported to ONNX: {model_path.replace('.pt', '.onnx')}")

        # Also export to TFLite for alternative deployment
        # model.export(format="tflite", imgsz=416)
        print("[i] Copy the .onnx file to Raspberry Pi for deployment")
    except Exception as e:
        print(f"[!] Export failed: {e}")


def validate_model(model_path: str, yaml_path: str):
    """Validate trained model on test set."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        metrics = model.val(data=yaml_path, imgsz=416)
        print(f"\n[✓] Validation Results:")
        print(f"    mAP50    : {metrics.box.map50:.4f}")
        print(f"    mAP50-95 : {metrics.box.map:.4f}")
        print(f"    Precision: {metrics.box.p.mean():.4f}")
        print(f"    Recall   : {metrics.box.r.mean():.4f}")
        return metrics
    except Exception as e:
        print(f"[!] Validation failed: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Plastic Waste Detector")
    parser.add_argument("--dataset", type=str, default="./dataset", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu", help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--roboflow-key", type=str, default="", help="Roboflow API key")
    args = parser.parse_args()

    # Step 1: Setup dataset
    dataset_path = args.dataset
    os.makedirs(dataset_path, exist_ok=True)

    if args.roboflow_key:
        print("[→] Downloading dataset from Roboflow...")
        # Replace these with your Roboflow project details
        download_dataset_roboflow(
            api_key=args.roboflow_key,
            workspace="your-workspace",
            project="plastic-waste-water",
            version=1,
            save_dir=dataset_path
        )

    # Step 2: Create YAML
    yaml_path = create_dataset_yaml(dataset_path)

    # Step 3: Train
    results = train_yolov8(
        yaml_path=yaml_path,
        model_size="yolov8n",
        epochs=args.epochs,
        device=args.device
    )

    print("\n[✓] Training complete!")
    print("    Best model: plastic_pollution/drone_v1/weights/best.pt")
    print("    ONNX model: plastic_pollution/drone_v1/weights/best.onnx")
    print("\nNext Step: Copy best.pt or best.onnx to Raspberry Pi")
    print("           Then run: python detect.py --model best.pt --source camera")
