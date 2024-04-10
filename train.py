from ultralytics import YOLO

# Load a model
# yaml会自动下载
model = YOLO(
    r"@C:\Users\CGW\ultralytics\ultralytics\cfg\models\v8\yolov8.yaml")  # build a new model from scratch
model = YOLO(r"@C:\Users\CGW\ultralytics\yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data=r"@C:\Users\CGW\ultralytics\ultralytics\cfg\datasets\coco128.yaml",
    epochs=100,
    imgsz=640)
