"""ขั้น 1: วัด layout จาก template + ทดสอบจัดภาพ
รัน:  python step1_check_align.py                 (ทดสอบทุกรูปใน Raw_Dataset)
      python step1_check_align.py path/to/img.jpg (ทดสอบรูปเดียว)
ผล:  out/align_check/*.jpg  — ภาพที่จัดแล้ว + ตาราง (ดูด้วยตาว่าตรงช่อง)
"""
import os, sys, cv2
from common import *

LAYOUT, REF = load_layouts()
for p in [1, 2, 3]:
    print(f"page{p}: rows={len(LAYOUT[p]['rows'])} header={list(LAYOUT[p]['header'])}")

paths = sys.argv[1:] or list_imgs(RAW_DIR)
d = os.path.join(OUT, "align_check"); os.makedirs(d, exist_ok=True)
ok = 0
for f in paths:
    rel = os.path.relpath(f, RAW_DIR) if f.startswith(RAW_DIR) else os.path.basename(f)
    try:
        aligned, page, L, H, rot, score = process_sheet(f)
    except Exception as e:
        print(f"  ✗ {rel}: {e}")
        continue
    ok += 1
    cells = sheet_cells(aligned, L)
    n_ink = sum(core.ink_ratio(c) >= INK_TH for fld, q, i, c in cells if fld == "ans")
    print(f"  ✓ {rel}: page={page} rot={rot} score={score:.2f} ช่องคำตอบที่มีหมึก={n_ink}/270")
    name = rel.replace(os.sep, "_").rsplit(".", 1)[0]
    cv2.imwrite(os.path.join(d, f"{name}.jpg"), cv2.resize(draw_layout(aligned, L), None, fx=0.5, fy=0.5))
print(f"\nจัดภาพสำเร็จ {ok}/{len(paths)} — ดูภาพที่ {d}")
