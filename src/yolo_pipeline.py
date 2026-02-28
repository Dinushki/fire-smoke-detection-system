from ultralytics import YOLO

# Load YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # nano version, fast & lightweight

# Run detection on a sample image
results = model("data/sample_fire.jpg")

# Show and save results
for r in results:
    r.show()       # opens window with detections
    r.save("outputs/")  # saves annotated image
