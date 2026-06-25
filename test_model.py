from ultralytics import YOLO

model = YOLO("model/best.pt")

results = model.predict(
    source="test_images",
    conf=0.25,
    save=True
)

print("Done!")