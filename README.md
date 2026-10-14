# ระบบอ่านตัวเลขเขียนมือบนกระดาษคำตอบจากภาพถ่าย

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/67160363/handwritten-answer-sheet/blob/main/demo.ipynb)

Term project วิชา Computer Vision: อ่านตัวเลขเขียนมือบนกระดาษคำตอบ (Template 3) จากรูปถ่ายมือถือ แล้วส่งออกเป็น JSON เพื่อตรวจคะแนน
Pipeline: Pre-processing (flatten แสง + จัดภาพด้วยจุดอ้างอิง) → ตำแหน่งแถว (layout / YOLOv8n) → Grid Cell Segmentation → Character Check → HWR ด้วย CNN ที่ propose เอง (**DigitNet-T3**, ~0.29M params) → JSON

แบบฟอร์มกระดาษคำตอบดัดแปลงมาจากโครงการ **Rayong MARCO** (ผลงานรุ่นพี่)

## ผลปัจจุบัน
- จัดภาพสำเร็จ 30/30 รูปถ่ายจริง (`step1_check_align.py`)
- DigitNet-T3 test accuracy 97.44% (11 คลาส, 3,432 ช่อง) — **ยังเป็นผลเบื้องต้น**: dataset มีเลขละ 3 แผ่น (train/val/test = 1/1/1) และคลาส blank มาจาก template ดิจิทัล
- ยังไม่ได้วัด CAR/WAR บนแผ่นที่เขียนตัวเลขผสม (baseline Template 3: CAR 98.80%, WAR 98.44%)

## ไฟล์
| ไฟล์ | หน้าที่ |
|---|---|
| `demo.ipynb` | โชว์ทุกขั้นตามสไลด์ พร้อมผลรันจริง |
| `t3core.py` | ฟังก์ชันภาพ: flatten, หาจุดอ้างอิง, วัด layout, จัดภาพ, แบ่งช่อง, ink ratio |
| `common.py` | path/config, อ่านรูป (HEIC+EXIF), โหลด layout, `process_sheet` |
| `digitnet.py` | โมเดล DigitNet-T3 + transforms |
| `step1_check_align.py` | ทดสอบจัดภาพทุกรูป → `out/align_check/` |
| `step2_make_dataset.py` | ตัดช่อง → dataset CNN (แบ่งตามแผ่น) + dataset YOLO (label อัตโนมัติ) |
| `step3_train_yolo.py` | เทรน YOLOv8n หาแถวคำตอบ/ช่องรหัส/วงกลม |
| `step4_train_cnn.py` | pre-train MNIST → fine-tune DigitNet-T3 → report + confusion matrix |
| `step5_predict.py` | อ่านแผ่นจริง → JSON, `--eval` คำนวณ CAR/WAR |
| `T3/` | template เปล่า 3 หน้า |
| `out/digitnet_t3.pt` | weights ที่เทรนแล้ว |

## การรัน
```bash
pip install -r requirements.txt
# วางรูปที่ data/Raw_Dataset/<0-9>/ (แต่ละแผ่นเขียนเลขเดียวกันทุกช่อง) — dataset ไม่ได้อยู่ใน repo
python step1_check_align.py
python step2_make_dataset.py
python step4_train_cnn.py --epochs 15
python step5_predict.py path/to/sheet.jpg
```

## Paper อ้างอิง
1. W. Yookwan et al., "Automated Optical Marker Exam Recognition using Cross-Linear Area Detection from Smartphone Camera," SNPD 2025. https://doi.org/10.1109/SNPD65828.2025.11253724
2. S. Mondal, P. De, S. Malakar, R. Sarkar, "OMRNet: A lightweight deep learning model for optical mark recognition," *Multimedia Tools and Applications* 83, 2024. https://doi.org/10.1007/s11042-023-15408-8
3. M. Li et al., "TrOCR: Transformer-Based Optical Character Recognition with Pre-trained Models," AAAI 2023. https://arxiv.org/abs/2109.10282
4. K. Ma, Z. Shu, X. Bai, J. Wang, D. Samaras, "DocUNet: Document Image Unwarping via a Stacked U-Net," CVPR 2018. https://doi.org/10.1109/CVPR.2018.00494
5. D. Prasad et al., "CascadeTabNet: An Approach for End to End Table Detection and Structure Recognition from Image-based Documents," CVPRW 2020. https://doi.org/10.1109/CVPRW50498.2020.00294
