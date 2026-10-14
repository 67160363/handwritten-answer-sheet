"""ขั้น 5: ตรวจข้อสอบจริง -> JSON (+ CAR/WAR ถ้ามีเฉลย)
รัน:  python step5_predict.py img1.jpg [img2.jpg ...]   -> พิมพ์ JSON
      python step5_predict.py --eval                    -> ประเมินทุกรูปใน data/eval (ต้องมี .csv เฉลยชื่อเดียวกัน)
เฉลย CSV:  field,value   (student_id,65123456 / q1,12.5 / q2,3 ...)
ผล:  out/results.json
"""
import os, sys, json, difflib
import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F
from common import *

YOLO_CLASSES = ["answer_row", "student_id", "subject_id", "school_id", "quiz_set", "page_no"]
YOLO_W = 960
CONF_TH = 0.6                      # CNN มั่นใจต่ำกว่านี้ -> ติดธงให้คนตรวจซ้ำ
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- โหลด CNN
from digitnet import DigitNet, plain
model = DigitNet().to(dev)
model.load_state_dict(torch.load(os.path.join(OUT, "digitnet_t3.pt"), map_location=dev)); model.eval()

# ---- โหลด YOLO (ไม่มีก็ใช้พิกัด layout แทน)
yolo_path = os.path.join(OUT, "yolo_runs", "t3", "weights", "best.pt")
yolo = None
if os.path.exists(yolo_path):
    from ultralytics import YOLO
    yolo = YOLO(yolo_path)
else:
    print("⚠ ไม่พบ YOLO best.pt — ใช้พิกัดจาก layout อย่างเดียว")


@torch.no_grad()
def read_cells(cells):
    out, batch, idx = [], [], []
    for k, c in enumerate(cells):
        if core.ink_ratio(c) < INK_TH:
            out.append(("", 1.0))                       # Character Check: ว่าง
        else:
            out.append(None); idx.append(k); batch.append(plain(core.prep_cell(c, 32)))
    if batch:
        pr = F.softmax(model(torch.stack(batch).to(dev)), 1).cpu()
        for k, p in zip(idx, pr):
            j = int(p.argmax())
            out[k] = ("" if CLASSES[j] == "blank" else CLASSES[j], float(p[j]))
    return out


def yolo_boxes(aligned):
    if yolo is None:
        return []
    s = YOLO_W / core.STD_W
    r = yolo(cv2.resize(aligned, None, fx=s, fy=s), imgsz=YOLO_W, conf=0.4, verbose=False)[0]
    return [(int(c), (b / s).tolist()) for c, b in zip(r.boxes.cls.cpu(), r.boxes.xyxy.cpu())]


def match(target, boxes, cls, tol=40):
    """box ของ YOLO ที่ใกล้กรอบ layout ที่สุด; ไม่เจอ -> ใช้กรอบ layout"""
    tx, ty = target[0] + target[2] / 2, target[1] + target[3] / 2
    best = None
    for c, (x1, y1, x2, y2) in boxes:
        d = abs((x1 + x2) / 2 - tx) + abs((y1 + y2) / 2 - ty)
        if c == cls and d < tol and (best is None or d < best[0]):
            best = (d, [x1 + 6, y1 + 6, x2 - x1 - 12, y2 - y1 - 12])   # ตัด pad 6px ตอนทำ label
    return (best[1], True) if best else (target, False)


def read_sheet(path):
    aligned, page, L, H, rot, score = process_sheet(path)
    g = core.flatten(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
    boxes = yolo_boxes(aligned)
    res = {"file": os.path.basename(path), "page": page, "align_score": round(score, 3),
           "quiz_set": core.bubble_filled(aligned, L["quiz_set"])[0], "answers": {}, "flags": []}
    for name, r in L["header"].items():
        rect, _ = match(r, boxes, YOLO_CLASSES.index(name))
        res[name] = "".join(c for c, _ in read_cells(core.split_row(g, rect, n=core.HEADER_CELLS)))
    n_yolo = 0
    for q, r in enumerate(L["rows"]):
        rect, hit = match(r, boxes, 0); n_yolo += hit
        cells = core.split_row(g, rect)
        chars = read_cells(cells[:core.DECIMAL_IDX] + cells[core.DECIMAL_IDX + 1:])   # ข้ามช่องจุดทศนิยม
        intp = "".join(c for c, _ in chars[:core.DECIMAL_IDX])
        decp = "".join(c for c, _ in chars[core.DECIMAL_IDX:])
        qn = (page - 1) * 30 + q + 1
        res["answers"][qn] = (intp or ("0" if decp else "")) + ("." + decp if decp else "")
        if min(p for _, p in chars) < CONF_TH:
            res["flags"].append(qn)
    res["yolo_rows_found"] = n_yolo
    return res


def car_war(pairs):
    """CAR = ตัวอักษรถูก/ทั้งหมด, WAR = ฟิลด์ถูกทั้งฟิลด์/ทั้งหมด"""
    ch_ok = ch_all = w_ok = 0
    for gt, pr in pairs:
        sm = difflib.SequenceMatcher(None, gt, pr)
        ch_ok += sum(b.size for b in sm.get_matching_blocks()); ch_all += max(len(gt), len(pr), 1)
        w_ok += gt == pr
    return ch_ok / ch_all, w_ok / len(pairs)


if __name__ == "__main__":
    results = []
    if "--eval" in sys.argv:
        pairs = []
        for f in list_imgs(EVAL_DIR):
            k = os.path.splitext(f)[0] + ".csv"
            if not os.path.exists(k):
                continue
            gt = {r.field: ("" if pd.isna(r.value) else str(r.value).strip())
                  for r in pd.read_csv(k, dtype=str).itertuples()}
            r = read_sheet(f); results.append(r)
            for fld, v in gt.items():
                pr = r["answers"].get(int(fld[1:]), "") if fld.startswith("q") else r.get(fld, "")
                pairs.append((v, pr))
                if v != pr:
                    print(f"  ✗ {r['file']} {fld}: เฉลย={v!r} อ่านได้={pr!r}")
        if pairs:
            car, war = car_war(pairs)
            print(f"\nCAR = {car:.4f}  WAR = {war:.4f}   (baseline T3: CAR 0.9880, WAR 0.9844)")
        else:
            print("ไม่พบรูป+เฉลยใน", EVAL_DIR)
    else:
        for f in sys.argv[1:]:
            r = read_sheet(f); results.append(r)
            print(json.dumps(r, ensure_ascii=False, indent=1))
    json.dump(results, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
