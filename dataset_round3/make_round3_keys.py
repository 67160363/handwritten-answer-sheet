"""Dataset รอบ 3: คนเขียนใหม่ 10 คน (W6-W15) x 10 แผ่น = 100 แผ่น  หน้า 1-3 (เทรนเฉพาะหน้า 1 = ข้อ 1-30)
ตัวเลขสุ่มแบบมีโครงสร้าง:
  40% แบบคำตอบข้อสอบจริง (12, 3.5, 0.25, 144)
  25% ยาวเต็มช่อง (83619.307)
  20% คู่ที่สับสนง่าย (1/7, 4/9, 3/8, 5/6, 0/6 ...)
  15% สั้น 1-2 หลัก
และเลือกคำตอบให้เลข 0-9 ออกพอๆ กัน
รัน (จากโฟลเดอร์โปรเจกต์):  python dataset_round3/make_round3_keys.py
"""
import os, csv, json, random, sys
import cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.getcwd())
import t3core as core

OUT = "dataset_round3"
WRITERS = [f"W{i}" for i in range(6, 16)]          # W6..W15 (ไม่ชนกับ W1-W5 รอบ 2)
PAGE_CYCLE = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]        # ต่อคน: หน้า1 x4, หน้า2 x3, หน้า3 x3 (เพื่อนเขียนชุดนี้แล้ว ห้ามเปลี่ยน)
TRAIN_PAGES = [1]                                  # เทรนเฉพาะข้อ 1-30 (แผ่น 01-04 ของแต่ละคน)
SEED = 2027
TPL = {1: "T3/template3_page1_role.jpg", 2: "T3/template3_page2.jpg", 3: "T3/template3_page3.jpg"}
FONT_D = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_TH = "/usr/share/fonts/opentype/tlwg/Loma.otf"
FONT_THB = "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf"
for f in [FONT_D, FONT_TH, FONT_THB]:   # Windows: ใช้ฟอนต์ระบบแทน
    if not os.path.exists(f):
        FONT_D = FONT_TH = FONT_THB = "C:/Windows/Fonts/tahoma.ttf"
        break

rng = random.Random(SEED)
digit_count = {str(d): 0 for d in range(10)}
PAIRS = [("1", "7"), ("4", "9"), ("3", "8"), ("5", "6"), ("0", "6"), ("1", "4"), ("2", "7"), ("0", "8"), ("5", "3")]


def digits(n, pool="0123456789", lead_nonzero=False):
    s = [rng.choice(pool) for _ in range(n)]
    if lead_nonzero and n > 1 and s[0] == "0":
        s[0] = rng.choice([c for c in pool if c != "0"] or ["1"])
    return "".join(s)


def gen(kind):
    if kind == "real":
        L = rng.choices([1, 2, 3, 4], [35, 35, 20, 10])[0]
        D = rng.choices([0, 1, 2, 3], [45, 30, 20, 5])[0]
        ip = digits(L, lead_nonzero=True)
        dp = digits(D)
        if dp and rng.random() < 0.3:
            dp = dp[:-1] + rng.choice("05")
    elif kind == "long":
        ip = digits(rng.choice([5, 6]), lead_nonzero=True)
        dp = digits(rng.choice([2, 3]))
    elif kind == "pair":
        a, b = rng.choice(PAIRS)
        ip = digits(rng.choice([3, 4, 5, 6]), a + b, lead_nonzero=True)
        dp = digits(rng.choice([0, 1, 2, 3]), a + b)
    else:  # short
        ip = digits(rng.choice([1, 2]), lead_nonzero=True)
        dp = ""
    return ip, dp


def make_answer():
    kind = rng.choices(["real", "long", "pair", "short"], [40, 25, 20, 15])[0]
    # สุ่ม 4 ตัวเลือก แล้วเลือกตัวที่ใช้เลขที่ยังออกน้อย -> เลข 0-9 สมดุล
    best = None
    for _ in range(4):
        ip, dp = gen(kind)
        score = sum(1.0 / (1 + digit_count[c]) for c in ip + dp)
        if best is None or score > best[0]:
            best = (score, ip, dp)
    _, ip, dp = best
    for c in ip + dp:
        digit_count[c] += 1
    value = ip + ("." + dp if dp else "")
    cells = ip.rjust(6, "_") + "." + dp.ljust(3, "_")
    return value, cells, kind


def make_id(n):
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(n - 1))


os.makedirs(f"{OUT}/keys", exist_ok=True)
os.makedirs(f"{OUT}/guides_png", exist_ok=True)
fD = ImageFont.truetype(FONT_D, 42)
fTag = ImageFont.truetype(FONT_THB, 40)
layouts = {p: core.build_layout(cv2.imread(TPL[p])) for p in TPL}
rows_all, kinds = [], {}
writer_pages = {}

for wname in WRITERS:
    writer_pages[wname] = []
    for si, p in enumerate(PAGE_CYCLE, 1):
        sid = f"{wname}_{si:02d}"
        header = {"student_id": make_id(8), "subject_id": make_id(6), "school_id": make_id(10)}
        L = layouts[p]
        key = dict(header)
        cells_q = {}
        for q in range(30):
            v, c, k = make_answer()
            kinds[k] = kinds.get(k, 0) + 1
            qn = (p - 1) * 30 + q + 1
            key[f"q{qn}"] = v
            cells_q[qn] = c
        with open(f"{OUT}/keys/{sid}_p{p}.csv", "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp); w.writerow(["field", "value"])
            for k, v in key.items():
                w.writerow([k, v])
        for k, v in key.items():
            cells = cells_q[int(k[1:])] if k.startswith("q") else v.ljust(14, "_")
            rows_all.append([wname, sid, p, k, v, cells])

        img = Image.fromarray(cv2.cvtColor(core.to_std(cv2.imread(TPL[p])), cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(img)

        def put(rect, chars, n):
            x, y, w, h = rect
            cw = w / n
            for i, ch in enumerate(chars):
                if ch in "_.":
                    continue
                dr.text((x + (i + 0.5) * cw, y + h / 2), ch, font=fD, fill=(25, 60, 150), anchor="mm")

        for qi, r in enumerate(L["rows"]):
            put(r, cells_q[(p - 1) * 30 + qi + 1], 10)
        for name, r in L["header"].items():
            put(r, header[name].ljust(14, "_"), 14)
        dr.text((250, 145), f"{sid.replace('_', '-')}  หน้า {p}", font=fTag, fill=(190, 40, 40), anchor="lm")
        img.save(f"{OUT}/guides_png/{sid}_p{p}.png")
        writer_pages[wname].append(img)

with open(f"{OUT}/answer_keys_all.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp); w.writerow(["writer", "sheet", "page", "field", "value", "cells"])
    w.writerows(rows_all)

# ---- หน้าคำแนะนำ
W, H = writer_pages[WRITERS[0]][0].size
fH = ImageFont.truetype(FONT_THB, 64); fB = ImageFont.truetype(FONT_TH, 40); fBb = ImageFont.truetype(FONT_THB, 40)


def instructions(wname):
    ins = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(ins)
    lines = [
        (fH, f"วิธีเขียน Dataset รอบ 3  (ชุดของ {wname})"), (fB, ""),
        (fBb, "1. ปริ้นกระดาษเปล่า Template 3 รวม 10 แผ่น"),
        (fB, "    หน้า 1 (ข้อ 1-30) x 4 แผ่น · หน้า 2 (ข้อ 31-60) x 3 แผ่น · หน้า 3 (ข้อ 61-90) x 3 แผ่น"),
        (fBb, "2. ดูตัวเลขจากใบแนว แล้วเขียนลงช่องเดียวกันบนกระดาษเปล่า"),
        (fB, "    จำนวนเต็ม: ชิดขวา ติดช่องจุดทศนิยม · ทศนิยม: ชิดซ้าย ต่อจากจุด"),
        (fB, "    ช่องที่ใบแนวว่าง ให้เว้นว่าง (ห้ามเขียน 0 เติม)"),
        (fB, "    ช่องรหัสด้านบน: เขียนชิดซ้ายตามใบแนว"),
        (fBb, "3. เขียนเร็วๆ ด้วยลายมือปกติ เหมือนตอนทำข้อสอบจริง"),
        (fB, "    ไม่ต้องตั้งใจเขียนให้สวย · ใช้ปากกา/ดินสอของตัวเอง สีน้ำเงินหรือดำ"),
        (fBb, "4. เขียนรหัสแผ่นและเลขหน้า (เช่น " + wname + "-01 หน้า 1) ที่บรรทัดชื่อ"),
        (fBb, "5. ถ่ายรูปทั้งแผ่น ให้เห็นสี่เหลี่ยมดำครบทั้ง 3 มุม"),
        (fB, "    ส่งไฟล์ต้นฉบับจากกล้อง (ห้ามแคปหน้าจอ / ห้ามส่ง LINE แบบย่อ)"),
        (fB, f"    ตั้งชื่อไฟล์: {wname}_01_p1.jpg ... {wname}_10_p3.jpg (ดูเลขหน้าที่มุมใบแนว)"),
        (fBb, "6. เขียนคนเดียวทั้ง 10 แผ่น (ห้ามให้คนอื่นช่วยเขียน)"),
        (fB, ""),
        (fBb, "ถ้าเขียนผิด: ขีดฆ่าแล้วแจ้งรหัสแผ่น + เลขข้อ หรือเริ่มแผ่นใหม่"),
    ]
    y = 160
    for f, t in lines:
        d.text((120, y), t, font=f, fill=(20, 30, 45)); y += 78
    return ins


for wname, pages in writer_pages.items():
    ins = instructions(wname)
    ins.save(f"{OUT}/guides_png/{wname}_00_instructions.png")
    ins.save(f"{OUT}/guides_{wname}.pdf", save_all=True, append_images=pages, resolution=200)

# ---- สรุป
lens = {}
for _, s, p, k, v, c in rows_all:
    if k.startswith("q"):
        ip, _, dp = v.partition(".")
        lens.setdefault(("int", len(ip)), 0); lens[("int", len(ip))] += 1
        lens.setdefault(("dec", len(dp)), 0); lens[("dec", len(dp))] += 1
n_sheets = len(WRITERS) * len(PAGE_CYCLE)
summary = {"writers": WRITERS, "sheets": n_sheets, "answers": n_sheets * 30,
           "pages": {p: PAGE_CYCLE.count(p) * len(WRITERS) for p in (1, 2, 3)},
           "answer_kinds": kinds,
           "digits_total": sum(digit_count.values()), "digit_count": digit_count,
           "int_len": {k[1]: v for k, v in sorted(lens.items()) if k[0] == "int"},
           "dec_len": {k[1]: v for k, v in sorted(lens.items()) if k[0] == "dec"}}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
