"""ค่าตั้งและฟังก์ชันที่ทุกสคริปต์ใช้ร่วมกัน"""
import os, glob, json
import cv2, numpy as np
import t3core as core

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.environ.get("RAW_DIR", os.path.join(ROOT, "data", "Raw_Dataset"))  # 0/ … 9/ blank/
TEMPLATE_DIR = os.environ.get("TEMPLATE_DIR", os.path.join(ROOT, "T3"))         # template เปล่า page1/2/3
EVAL_DIR = os.path.join(ROOT, "data", "eval")                                    # แผ่นทดสอบ + เฉลย .csv
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

CLASSES = [str(i) for i in range(10)] + ["blank"]
INK_TH = 0.04          # ink ratio ต่ำกว่านี้ = ช่องว่าง (Character Check)
IMG_EXT = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp")


def list_imgs(folder):
    """รูปทุกไฟล์ในโฟลเดอร์ (รวมโฟลเดอร์ย่อย)"""
    return sorted(f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                  if f.lower().endswith(IMG_EXT))


def imread(path):
    """อ่าน jpg/png/HEIC + หมุนตาม EXIF (cv2.imread ไม่สน EXIF)"""
    from PIL import Image, ImageOps
    if path.lower().endswith((".heic", ".heif")):
        import pillow_heif
        pillow_heif.register_heif_opener()
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def template_path(p):
    fs = [f for f in list_imgs(TEMPLATE_DIR) if f"page{p}" in os.path.basename(f)]
    if not fs:
        raise FileNotFoundError(f"ไม่พบ template page{p} ใน {TEMPLATE_DIR}")
    return fs[0]


_LAYOUT, _REF = {}, {}


def load_layouts():
    """layout ของ 3 หน้า (cache ไว้ที่ out/layout_p*.json)"""
    if _LAYOUT:
        return _LAYOUT, _REF
    for p in [1, 2, 3]:
        tpl = imread(template_path(p))
        cache = os.path.join(OUT, f"layout_p{p}.json")
        if os.path.exists(cache):
            L = json.load(open(cache))
        else:
            L = core.build_layout(tpl)
            assert len(L["rows"]) == 30 and len(L["header"]) == 3, f"page{p}: หา layout ไม่ครบ"
            json.dump(L, open(cache, "w"))
        _LAYOUT[p] = L
        _REF[p] = cv2.cvtColor(core.to_std(tpl), cv2.COLOR_BGR2GRAY)
    return _LAYOUT, _REF


def process_sheet(path):
    """รูปถ่าย -> (aligned, page, layout, H, rot, score)"""
    LAYOUT, REF = load_layouts()
    aligned, H, rot, score = core.align(imread(path), LAYOUT[1], [REF[p] for p in (1, 2, 3)])
    page, _ = core.bubble_filled(aligned, LAYOUT[1]["page_no"])
    page = page or 1
    return aligned, page, LAYOUT[page], H, rot, score


def sheet_cells(aligned, L):
    """ทุกช่องในหน้า -> list ของ (field, q, idx, cell_gray) ข้ามช่องจุดทศนิยม"""
    g = core.flatten(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
    out = []
    for q, r in enumerate(L["rows"]):
        for i, c in enumerate(core.split_row(g, r)):
            if i != core.DECIMAL_IDX:
                out.append(("ans", q, i, c))
    for name, r in L["header"].items():
        for i, c in enumerate(core.split_row(g, r, n=core.HEADER_CELLS)):
            out.append((name, 0, i, c))
    return out


def draw_layout(img, L):
    a = img.copy()
    g = core.flatten(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    for r in L["rows"]:
        x, y, w, h = core.refine_rect(g, r)
        cv2.rectangle(a, (x, y), (x + w, y + h), (0, 0, 255), 3)
        for k in range(1, 10):
            xx = int(x + k * w / 10)
            cv2.line(a, (xx, y), (xx, y + h), (255, 0, 0), 1)
    for r in L["header"].values():
        x, y, w, h = core.refine_rect(g, r)
        cv2.rectangle(a, (x, y), (x + w, y + h), (0, 160, 0), 3)
        for k in range(1, 14):
            xx = int(x + k * w / 14)
            cv2.line(a, (xx, y), (xx, y + h), (255, 0, 0), 1)
    for x, y, w, h in L["quiz_set"] + L["page_no"]:
        cv2.rectangle(a, (x, y), (x + w, y + h), (255, 0, 255), 3)
    return a
