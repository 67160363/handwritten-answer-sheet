"""ขั้น 2: สร้าง dataset ช่องตัวเลข (CNN) + dataset YOLO (label อัตโนมัติจาก layout)
Raw_Dataset/<เลข>/ = แผ่นที่ทุกช่องเขียนเลขเดียวกัน, Raw_Dataset/blank/ = แผ่นเปล่า
ผล:  out/cells/<split>/<class>/*.png, out/cells.csv, out/yolo/{images,labels}/{train,val}, out/yolo/data.yaml
"""
import os, csv, random, shutil
from collections import Counter
import numpy as np, cv2
from common import *

CELL_DIR = os.path.join(OUT, "cells")
YOLO_DIR = os.path.join(OUT, "yolo")
YOLO_CLASSES = ["answer_row", "student_id", "subject_id", "school_id", "quiz_set", "page_no"]
YOLO_W = 960
shutil.rmtree(CELL_DIR, ignore_errors=True)
shutil.rmtree(YOLO_DIR, ignore_errors=True)

# ---- แบ่ง split ตาม "แผ่น" (ห้ามแบ่งตามช่อง: ลายมือแผ่นเดียวกันจะรั่วข้าม train/test)
# คลาสที่มี < 3 แผ่น (ช่วงทดลอง) -> แบ่งตามข้อ: q1-20 train, 21-25 val, 26-30 test
#   ใช้ทดสอบ pipeline ได้ แต่ accuracy สูงเกินจริง ห้ามใช้รายงานผล
random.seed(42)
sheets = []
for lab in CLASSES:
    fs = list_imgs(os.path.join(RAW_DIR, lab))
    random.shuffle(fs)
    n = len(fs)
    n_te = n_va = max(1, round(0.15 * n))
    for j, f in enumerate(fs):
        if n < 3:
            sp = "by_row"
        else:
            sp = "test" if j < n_te else ("val" if j < n_te + n_va else "train")
        sheets.append((f, lab, sp))
print("จำนวนแผ่น:", dict(Counter(s[1] for s in sheets)))
print("วิธีแบ่ง  :", dict(Counter(s[2] for s in sheets)))


def row_split(field, q):
    if field != "ans":
        return "train"
    q = q % 30
    return "train" if q < 20 else ("val" if q < 25 else "test")


def union(rs):
    x0 = min(r[0] for r in rs); y0 = min(r[1] for r in rs)
    return [x0, y0, max(r[0] + r[2] for r in rs) - x0, max(r[1] + r[3] for r in rs) - y0]


def yolo_labels(L, pad=6):
    objs = [(0, r) for r in L["rows"]]
    objs += [(YOLO_CLASSES.index(k), r) for k, r in L["header"].items()]
    objs += [(4, union(L["quiz_set"])), (5, union(L["page_no"]))]
    lines = []
    for c, (x, y, w, h) in objs:
        x, y, w, h = x - pad, y - pad, w + 2 * pad, h + 2 * pad
        lines.append(f"{c} {(x + w/2)/core.STD_W:.6f} {(y + h/2)/core.STD_H:.6f} "
                     f"{w/core.STD_W:.6f} {h/core.STD_H:.6f}")
    return "\n".join(lines)


rows, inks, fails = [], [], []
for f, lab, split in sheets:
    try:
        aligned, page, L, *_ = process_sheet(f)
    except Exception as e:
        fails.append((f, str(e))); continue
    sid = f"{lab}_{os.path.splitext(os.path.basename(f))[0]}"
    # --- CNN cells
    for field, q, i, c in sheet_cells(aligned, L):
        ink = core.ink_ratio(c); inks.append((ink, lab))
        y = "blank" if (lab == "blank" or ink < INK_TH) else lab
        sp = row_split(field, q) if split == "by_row" else split
        d = os.path.join(CELL_DIR, sp, y); os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, f"{sid}_p{page}_{field}_q{q+1:02d}_c{i}.png")
        cv2.imwrite(fn, core.prep_cell(c, 32))
        rows.append([os.path.relpath(fn, ROOT), y, sid, sp])
    # --- YOLO (บนภาพที่จัดแล้ว)
    ysp = {"test": "val", "by_row": "val" if random.random() < 0.2 else "train"}.get(split, split)
    for k in ["images", "labels"]:
        os.makedirs(os.path.join(YOLO_DIR, k, ysp), exist_ok=True)
    s = YOLO_W / core.STD_W
    cv2.imwrite(os.path.join(YOLO_DIR, "images", ysp, f"{sid}.jpg"), cv2.resize(aligned, None, fx=s, fy=s))
    open(os.path.join(YOLO_DIR, "labels", ysp, f"{sid}.txt"), "w").write(yolo_labels(L))

with open(os.path.join(OUT, "cells.csv"), "w", newline="") as fp:
    csv.writer(fp).writerows([["path", "label", "sheet", "split"]] + rows)
open(os.path.join(YOLO_DIR, "data.yaml"), "w").write(
    f"path: {YOLO_DIR}\ntrain: images/train\nval: images/val\nnames: {YOLO_CLASSES}\n")

print(f"\nช่องทั้งหมด {len(rows)} | แผ่นที่จัดภาพไม่ได้ {len(fails)}")
for x in fails:
    print("   ✗", x)
for sp in ["train", "val", "test"]:
    print(f"  {sp:5s}", dict(sorted(Counter(r[1] for r in rows if r[3] == sp).items())))

# montage ตรวจ label ด้วยตา: 16 ตัวอย่างต่อคลาส
random.seed(0)
tiles = []
for lab in CLASSES:
    fs = [os.path.join(ROOT, r[0]) for r in rows if r[1] == lab and r[3] == "train"]
    pick = random.sample(fs, min(16, len(fs)))
    row = [cv2.resize(cv2.imread(p, 0), (48, 48), interpolation=cv2.INTER_NEAREST) for p in pick]
    row += [np.zeros((48, 48), np.uint8)] * (16 - len(row))
    tiles.append(np.hstack(row))
cv2.imwrite(os.path.join(OUT, "cells_montage.png"), np.vstack(tiles))
print("ดูตัวอย่างช่องแต่ละคลาส (แถวบนสุด = 0 ... ล่างสุด = blank):", os.path.join(OUT, "cells_montage.png"))
