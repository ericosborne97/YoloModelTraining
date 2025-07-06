from ultralytics import YOLO
m = YOLO("yolov10n.pt")   # or .yaml
print("model ok:", m.model.model.__class__.__name__)
