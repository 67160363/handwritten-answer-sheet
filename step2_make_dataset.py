"""ขั้น 2: สร้าง dataset ช่องตัวเลข (CNN) + dataset YOLO (label อัตโนมัติจาก layout)
รอบ 1: Raw_Dataset/0–9/<เลข>/ = แผ่นที่ทุกช่องเขียนเลขเดียวกัน, Raw_Dataset/blank/ = แผ่นเปล่า
รอบ 2: Raw_Dataset/mixed/<writer>/ = แผ่นตัวเลขผสม, label จาก dataset_round2/answer_keys_all.csv (+ corrections.csv)
รอบ 3: Raw_Dataset/mixed/W6..W15/ เหมือนรอบ 2 แต่ใช้ dataset_round3/ (เขียนบนกระดาษหน้า 1 ทุกแผ่น)
รัน:  python step2_make_dataset.py [--test-writer W5]
ผล:  out/cells/<split>/<class>/*.png, out/cells.csv, out/label_check.csv,
      out/yolo/{images,labels}/{train,val}, out/yolo/data.yaml
"""
import os, csv, random, shutil, argparse
from collections import Counter, defaultdict
import numpy as np, cv2
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--test-writer", default="W5", help="ทุกแผ่นของคนนี้ (รอบ 2) = test")
args = ap.parse_args()

CELL_DIR = os.path.join(OUT, "cells")
YOLO_DIR = os.path.join(OUT, "yolo")
YOLO_CLASSES = ["answer_row", "student_id", "subject_id", "school_id", "quiz_set", "page_no"]
YOLO_W = 960
R1_DIR = os.path.join(RAW_DIR, "0–9") if os.path.isdir(os.path.join(RAW_DIR, "0–9")) else RAW_DIR
R2_DIR = os.path.join(RAW_DIR, "mixed")
KEY_DIR = os.path.join(ROOT, "dataset_round2")
N_VAL_R2 = 2           # รอบ 2: จำนวนแผ่น val ต่อคน (ที่ไม่ใช่ test writer)
MAX_MISMATCH = 5       # label กับหมึกขัดกันเกินนี้ต่อแผ่น -> เตือน (อาจสลับแถว/label ผิด)
shutil.rmtree(CELL_DIR, ignore_errors=True)
shutil.rmtree(YOLO_DIR, ignore_errors=True)

# sheets: dict(path, sid, writer, round, split, labels)   labels=None -> label จากชื่อโฟลเดอร์ (รอบ 1)
sheets = []

# ---- รอบ 1: แบ่งตาม "กลุ่ม writer" (ห้ามแบ่งตามช่อง/แผ่น: ลายมือคนเดียวกันจะรั่วข้าม split)
# สุ่ม 1 กลุ่ม = val ที่เหลือ = train; ไม่มี test (test = test writer ของรอบ 2 อย่างเดียว)
r1 = []
for lab in CLASSES:
    for f in list_imgs(os.path.join(RAW_DIR if lab == "blank" else R1_DIR, lab)):
        name = os.path.splitext(os.path.basename(f))[0]
        writer = "r1_template" if lab == "blank" else "r1_" + name.replace("_", "")[:5]   # IMG_0439 -> r1_IMG04
        r1.append((f, lab, name, writer))
r1_val = random.Random(42).choice(sorted({w for *_, w in r1} - {"r1_template"}))
for f, lab, name, writer in r1:
    sp = "val" if writer == r1_val else "train"
    sheets.append(dict(path=f, sid=f"{lab}_{name}", writer=writer, round=1, split=sp, labels=None, lab=lab))


# ---- รอบ 2: แบ่งตาม "คนเขียน" — test writer ทั้งหมด = test, ที่เหลือสุ่ม N_VAL_R2 แผ่นต่อคน = val
def read_csv(name):
    with open(os.path.join(KEY_DIR, name), encoding="utf-8-sig") as fp:
        return list(csv.DictReader(fp))


def cell_labels(cells):
    return ["blank" if ch == "_" else ch for ch in cells]


keys = defaultdict(dict)                        # sheet -> {field: cells}
key_page = {}
by_writer = defaultdict(list)
# รอบ 2 = dataset_round2, รอบ 3 = dataset_round3 (W6-W15)
# รอบ 3: เพื่อนเขียนทุกแผ่นบนกระดาษ "หน้า 1" (ข้อ 1-30) แม้เฉลยเดิมจะเป็นข้อ 31-90
#        -> เลขข้อ 31-60 / 61-90 ถูกแมปกลับเป็นข้อ 1-30 ตามแถวที่เขียนจริง
# รอบ 4: dataset_other = แผ่น "Old" ของกลุ่มอื่น (writer OLD, หน้า 1, label จากกฎเลขเรียง) รูปอยู่ที่ CV project/Old
#        ช่อง label "?" = ไม่ใช้เทรน (เช่น รหัสนักศึกษาจริง), photo_map คอลัมน์ use=0 = ตัดทั้งแผ่น
for KD, ON_P1, IMG_DIR, PFX in [(KEY_DIR, False, R2_DIR, ""), (os.path.join(ROOT, "dataset_round3"), True, R2_DIR, ""),
                                (os.path.join(ROOT, "dataset_other"), True, os.path.join(ROOT, "CV project"), "old_")]:
    if not os.path.isdir(KD):
        continue
    rd = lambda n: list(csv.DictReader(open(os.path.join(KD, n), encoding="utf-8-sig"))) \
        if os.path.exists(os.path.join(KD, n)) else []
    for r in rd(PFX + "keys.csv" if PFX else "answer_keys_all.csv"):
        fld = r["field"]
        if ON_P1 and fld.startswith("q"):
            fld = f"q{(int(fld[1:]) - 1) % 30 + 1}"
        keys[r["sheet"]][fld] = r["cells"]
        key_page[r["sheet"]] = 1 if ON_P1 else int(r["page"])
    for r in rd(PFX + "corrections.csv"):        # row = เลขข้อ (ตามแถวบนกระดาษ) หรือชื่อ field header
        keys[r["sheet"]][f"q{int(r['row'])}" if r["row"].isdigit() else r["row"]] = r["cells"]
    for r in rd(PFX + "photo_map.csv"):
        if r.get("use", "1") == "0":
            continue
        f = os.path.join(IMG_DIR, r["writer"].title() if PFX else r["writer"], r["file"])   # OLD -> CV project/Old
        if not os.path.exists(f):
            print(f"  ⚠ ข้าม {r['sheet']}: ไม่มีไฟล์ {os.path.relpath(f, ROOT)}")
            continue
        by_writer[r["writer"]].append((r["sheet"], f))
if args.test_writer not in by_writer:
    raise SystemExit(f"--test-writer {args.test_writer} ไม่มีในรอบ 2 (มี {sorted(by_writer)})")
rng = random.Random(42)
for w in sorted(by_writer):
    ss = sorted(by_writer[w])
    val = set() if w == args.test_writer else set(s for s, _ in rng.sample(ss, N_VAL_R2))
    for s, f in ss:
        sp = "test" if w == args.test_writer else ("val" if s in val else "train")
        lab = {}
        for fld, cells in keys[s].items():
            if fld.startswith("q"):
                lab.update({("ans", int(fld[1:]) - 1, i): y for i, y in enumerate(cell_labels(cells))
                            if i != core.DECIMAL_IDX})
            else:
                lab.update({(fld, 0, i): y for i, y in enumerate(cell_labels(cells))})
        sheets.append(dict(path=f, sid=s, writer=w, round=2, split=sp, labels=lab))

print("จำนวนแผ่นต่อ writer:", dict(Counter(s["writer"] for s in sheets)))
print("วิธีแบ่ง (แผ่น)    :", dict(Counter((s["round"], s["split"]) for s in sheets)))


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


rows, fails, checks = [], [], []
for S in sheets:
    try:
        aligned, page, L, *_ = process_sheet(S["path"])
    except Exception as e:
        fails.append((S["path"], str(e))); continue
    sid, split = S["sid"], S["split"]
    if S["round"] == 2 and page != key_page.get(sid, page):
        print(f"  ⚠ {sid}: ตรวจได้หน้า {page} แต่เฉลยเป็นหน้า {key_page[sid]}")
    # --- CNN cells
    for field, q, i, c in sheet_cells(aligned, L):
        ink = core.ink_ratio(c)
        if S["labels"] is None:
            y = "blank" if (S["lab"] == "blank" or ink < INK_TH) else S["lab"]
        else:
            y = S["labels"].get((field, q, i), "blank")
            if y == "?":                                    # ไม่มี label ที่เชื่อได้ -> ไม่ใช้เทรน
                continue
            if (y == "blank") != (ink < INK_TH):          # label กับหมึกขัดกัน: เก็บไว้ใน dataset แต่บันทึก
                checks.append([sid, field, q + 1, i, y, f"{ink:.4f}"])
        d = os.path.join(CELL_DIR, split, y); os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, f"{sid}_p{page}_{field}_q{q+1:02d}_c{i}.png")
        cv2.imwrite(fn, core.prep_cell(c, 32))
        rows.append([os.path.relpath(fn, ROOT), y, sid, split, S["writer"], S["round"]])
    # --- YOLO (บนภาพที่จัดแล้ว)
    ysp = "val" if split == "test" else split
    for k in ["images", "labels"]:
        os.makedirs(os.path.join(YOLO_DIR, k, ysp), exist_ok=True)
    s = YOLO_W / core.STD_W
    cv2.imwrite(os.path.join(YOLO_DIR, "images", ysp, f"{sid}.jpg"), cv2.resize(aligned, None, fx=s, fy=s))
    open(os.path.join(YOLO_DIR, "labels", ysp, f"{sid}.txt"), "w").write(yolo_labels(L))

with open(os.path.join(OUT, "cells.csv"), "w", newline="") as fp:
    csv.writer(fp).writerows([["path", "label", "sheet", "split", "writer", "round"]] + rows)
with open(os.path.join(OUT, "label_check.csv"), "w", newline="", encoding="utf-8") as fp:
    csv.writer(fp).writerows([["sheet", "field", "q", "cell", "label", "ink"]] + checks)
open(os.path.join(YOLO_DIR, "data.yaml"), "w").write(
    f"path: {YOLO_DIR}\ntrain: images/train\nval: images/val\nnames: {YOLO_CLASSES}\n")

print(f"\nช่องทั้งหมด {len(rows)} | แผ่นที่จัดภาพไม่ได้ {len(fails)}")
for x in fails:
    print("   ✗", x)
for sp in ["train", "val", "test"]:
    print(f"  {sp:5s}", dict(sorted(Counter(r[1] for r in rows if r[3] == sp).items())))

n_chk = Counter(c[0] for c in checks)
print(f"\nlabel ขัดกับหมึก (รอบ 2) {len(checks)} ช่อง ใน {len(n_chk)} แผ่น -> {os.path.join(OUT, 'label_check.csv')}")
for sid, n in sorted(n_chk.items()):
    print(f"  {'⚠' if n > MAX_MISMATCH else ' '} {sid}: {n}")

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
