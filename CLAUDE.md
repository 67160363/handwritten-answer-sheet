# Rayong MARCO — Template 3 (Computer Vision Term Project)

## โจทย์ (จากสไลด์อาจารย์ `PROJECT-COMPUTER VISION.pdf`)
เขียนโปรแกรม **object detection and recognition** อ่านกระดาษคำตอบ Template 3
1. สร้าง dataset เอง  2. **Propose CNN model เอง**  3. **Accuracy > 90%**
Baseline ที่ต้องเทียบ (Template 3): **CAR 98.80%, WAR 98.44%, F1 99.63%**
- CAR = ความแม่นรายตัวอักษร, WAR = ความแม่นทั้งฟิลด์ (คำตอบทั้งข้อถูกครบ)
Flow อาจารย์: Pre-processing → YOLO (header + answer rows) → Grid Cell Segmentation (C0–C9) → Character Check (blank?) → HWR → JSON → Scoring
Paper อ้างอิงใน `Example_paper/` (เก็บในเครื่องเท่านั้น — gitignore เพราะลิขสิทธิ์, repo เป็น public; ลิงก์อยู่ใน README): CLA (ม.บูรพา, ใกล้สุด), OMRNet, TrOCR, DocUNet, CascadeTabNet

## แบบฟอร์ม Template 3
- 3 หน้า × 30 ข้อ = 90 ข้อ; แต่ละข้อ 10 ช่องเขียนตัวเลขลายมือ: 6 ช่อง + **ช่อง index 6 สีเทา = จุดทศนิยมพิมพ์ไว้แล้ว (ข้ามเสมอ)** + 3 ช่อง
- Header: รหัสประจำตัว / รหัสวิชา / รหัสสถานศึกษา แถวละ 14 ช่อง
- วงกลมฝน: ชุดกระดาษคำถาม (1-3), หน้ากระดาษคำตอบ (1-3) — ใช้รู้ว่าเป็นหน้าไหน
- จุดอ้างอิง: สี่เหลี่ยมดำล่างซ้าย/ล่างขวา, กล่องเลข "3" ขวาบน, ขีดดำข้างแถว (ขีดแถวแรกซ้าย = จุดที่ 4)
- Template เปล่า: `T3/template3_page{1,2,3}*.jpg`

## โครงโค้ด
| ไฟล์ | หน้าที่ |
|---|---|
| `t3core.py` | ฟังก์ชันภาพทั้งหมด: flatten แสง, หา marker, build_layout, align (ลองหมุน 0/180/90/270 + เช็ก score กับ template), refine_rect, split_row, ink_ratio, prep_cell |
| `common.py` | path/config, imread (HEIC+EXIF), load_layouts (cache `out/layout_p*.json`), process_sheet, sheet_cells |
| `digitnet.py` | **DigitNet-T3** (CNN ที่ propose): 3×[Conv-BN-ReLU×2+MaxPool] → GAP → Dropout → FC 11 คลาส, input 32×32, ~0.29M params |
| `step1_check_align.py` | ทดสอบจัดภาพทุกรูป → `out/align_check/` (ดูด้วยตาว่าตารางตรง) |
| `step2_make_dataset.py` | ตัดช่อง → `out/cells/`, `out/cells.csv`; สร้าง YOLO dataset + label อัตโนมัติจาก layout → `out/yolo/` |
| `step3_train_yolo.py` | YOLOv8n (6 คลาส: answer_row, student_id, subject_id, school_id, quiz_set, page_no) บนภาพที่จัดแล้ว |
| `step4_train_cnn.py` | (option) pre-train MNIST 3 epoch → fine-tune; รายงาน test → `out/cnn_report.txt`, confusion matrix |
| `demo.ipynb` | โชว์อาจารย์ตามลำดับสไลด์ import จากโมดูลเท่านั้น (ห้ามก็อปโค้ด) — แก้แล้วรัน `jupyter nbconvert --execute --inplace` |
| `step5_predict.py` | ตรวจจริง → JSON; `--eval` คำนวณ CAR/WAR จาก `data/eval/*.jpg` + `*.csv` (field,value) |

## ข้อมูล
- `data/Raw_Dataset/<0-9>/` = รูปถ่ายแผ่นที่ **ทุกช่องเขียนเลขเดียวกัน** → label จากชื่อโฟลเดอร์; ช่องที่ ink < `INK_TH` → คลาส `blank`
- `data/Raw_Dataset/blank/` = แผ่นเปล่า (ตอนนี้ใช้ template ดิจิทัล 3 ไฟล์)
- ต้นฉบับอยู่ใน Google Drive `MyDrive/Raw_Dataset` (ดาวน์โหลดมาวางที่ `data/`) — ตอนนี้มีแค่เลขละ 2 รูป
- เปลี่ยน path ได้ด้วย env `RAW_DIR`, `TEMPLATE_DIR`

## กฎสำคัญ (อย่าเปลี่ยนโดยไม่มีเหตุผล)
- **แบ่ง train/val/test ตาม "แผ่น" เท่านั้น** ห้ามสุ่มตามช่อง (ลายมือแผ่นเดียวกันรั่ว → accuracy สูงเกินจริง)
  - คลาสที่มี < 3 แผ่นใช้โหมด `by_row` ชั่วคราว (q1-20/21-25/26-30) — **ห้ามใช้ตัวเลขจากโหมดนี้รายงานผล**
- ทุกภาพต้อง align เป็นขนาดมาตรฐาน 1654×2339 ก่อนตัดช่อง (บั๊กเดิมจากโค้ด Gemini: ตัดที่ความละเอียดต้นฉบับแล้วกรองขนาดช่อง 15–80px → ได้ 0 รูป)
- YOLO: `fliplr=0` เสมอ (เอกสารกลับด้านไม่ได้)
- การเปลี่ยน threshold ใน `t3core.py` (`INK_LEVEL`, `DIGIT_LEVEL`, ขนาด marker) ต้องรัน step1 ซ้ำและดูภาพใน `out/align_check/` + `out/cells_montage.png`
- ตัวเลข/ข้อมูลที่จะใส่รายงานต้องมาจากผลรันจริงหรือ paper จริงเท่านั้น

## การรัน (Windows PowerShell)
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONUTF8=1          # ให้ print ภาษาไทยไม่พัง
python step1_check_align.py
python step2_make_dataset.py
python step4_train_cnn.py            # CNN เร็วบน CPU
python step3_train_yolo.py 60        # ช้าบน CPU — เทรนบน Colab ได้ แล้ววาง best.pt ที่ out/yolo_runs/t3/weights/
python step5_predict.py data/Raw_Dataset/0/xxx.jpg
```

## สถานะ
- [x] pipeline จัดภาพ/ตัดช่อง ทดสอบกับภาพจำลอง (เอียง, เงา, กลับหัว) และรูปจริงแผ่นเลข 0 (ภาพย่อ 529px) ผ่าน
- [x] step1 30/30, step2, step4 (15 epoch: test 97.44% — ผลเบื้องต้น เลขละ 3 แผ่น) รันแล้ว
- [ ] ยังไม่เคยรัน step3 (YOLO) และ step5 กับชุด eval
- [ ] เก็บข้อมูลเพิ่ม: เป้าหมายเลขละ 8–10 แผ่น หลายคน กระจาย 3 หน้า + แผ่นเปล่าที่ถ่ายจริง
- [ ] ทำชุด eval แผ่นที่เขียนตัวเลขผสมจริง + เฉลย CSV เพื่อวัด CAR/WAR เทียบ baseline
