"""ขั้น 3: เทรน YOLOv8n หาแถวคำตอบ/ช่องรหัส/วงกลม บนภาพที่จัดแล้ว
รัน:  python step3_train_yolo.py [epochs]
ผล:  out/yolo_runs/t3/weights/best.pt
หมายเหตุ: บน CPU ช้า — ถ้าช้าเกินไปเทรนบน Colab (GPU) แล้วเอา best.pt มาวางที่เดิม
"""
import os, sys
from ultralytics import YOLO
from common import OUT

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
model = YOLO("yolov8n.pt")
model.train(data=os.path.join(OUT, "yolo", "data.yaml"), imgsz=960, epochs=epochs, batch=8, patience=15,
            fliplr=0.0,            # เอกสารกลับซ้าย-ขวาไม่ได้
            mosaic=0.5, degrees=3, translate=0.05, scale=0.2, perspective=0.0005, hsv_v=0.4,
            project=os.path.join(OUT, "yolo_runs"), name="t3", exist_ok=True, workers=0)
m = YOLO(os.path.join(OUT, "yolo_runs", "t3", "weights", "best.pt")).val(
    data=os.path.join(OUT, "yolo", "data.yaml"), imgsz=960, workers=0)
print(f"mAP50 = {m.box.map50:.4f} | mAP50-95 = {m.box.map:.4f}")
