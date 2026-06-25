from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='./dataset/data.yaml',
    epochs=15,
    imgsz=320,
    batch=16,
    device='cpu',
    project='runs',
    name='plastic_v3',
    patience=5,
    augment=True,
    fliplr=0.5,
    degrees=15,
    hsv_s=0.7,
)

print('Done! Model at: runs/detect/plastic_v3/weights/best.pt')