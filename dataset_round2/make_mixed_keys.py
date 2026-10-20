"""สร้างชุดตัวเลขสุ่ม (คำตอบผสม) + เฉลย + ใบแนวให้คนลอก สำหรับเก็บ dataset รอบ 2"""
import os, csv, json, random
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    import t3core as core   # ในโปรเจกต์
except ImportError:
    import core

OUT = "dataset_round2"
N_WRITERS = 5
SHEETS_PER_WRITER = 10
PAGE_CYCLE = [1] * 10   # รอบนี้ทดลองเฉพาะหน้า 1 (ข้อ 1-30)
SEED = 2026
TPL = {1: "T3/template3_page1_role.jpg", 2: "T3/template3_page2.jpg", 3: "T3/template3_page3.jpg"}  # รันจากโฟลเดอร์โปรเจกต์
FONT_D = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_TH = "/usr/share/fonts/opentype/tlwg/Loma.otf"
FONT_THB = "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf"

rng = random.Random(SEED)
INT_LEN_W = {1: 18, 2: 18, 3: 18, 4: 16, 5: 15, 6: 15}      # หลักหน่วย..แสน
DEC_LEN_W = {0: 16, 1: 28, 2: 28, 3: 28}                       # ไม่มี/1/2/3 ตำแหน่ง


def pick(w):
    return rng.choices(list(w), weights=list(w.values()))[0]


def make_answer():
    L, D = pick(INT_LEN_W), pick(DEC_LEN_W)
    if L == 1:
        ip = str(rng.randint(0, 9))                     # อนุญาต 0.xx
    else:
        ip = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(L - 1))
    dp = "".join(str(rng.randint(0, 9)) for _ in range(D))
    value = ip + ("." + dp if dp else "")
    cells = ip.rjust(6, "_") + "." + dp.ljust(3, "_")  # จำนวนเต็มชิดขวา ทศนิยมชิดซ้าย
    return value, cells


def make_id(n):
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(n - 1))


os.makedirs(f"{OUT}/keys", exist_ok=True)
os.makedirs(f"{OUT}/guides_png", exist_ok=True)
fD = ImageFont.truetype(FONT_D, 42)
fTag = ImageFont.truetype(FONT_THB, 40)
layouts = {p: core.build_layout(cv2.imread(TPL[p])) for p in TPL}
rows_all, digit_count = [], {str(d): 0 for d in range(10)}

writer_pages = {}
for wi in range(1, N_WRITERS + 1):
  writer_pages[wi] = []
  for si, p in enumerate(PAGE_CYCLE, 1):
    sid = f"W{wi}_{si:02d}"
    header = {"student_id": make_id(8), "subject_id": make_id(6), "school_id": make_id(10)}
    if True:
        L = layouts[p]
        key = dict(header)
        cells_q = {}
        for q in range(30):
            v, c = make_answer()
            qn = (p - 1) * 30 + q + 1
            key[f"q{qn}"] = v
            cells_q[qn] = c
        # ---- เฉลยต่อแผ่น (รูปแบบเดียวกับ step5_predict --eval)
        with open(f"{OUT}/keys/{sid}_p{p}.csv", "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp); w.writerow(["field", "value"])
            for k, v in key.items():
                w.writerow([k, v])
        for k, v in key.items():
            cells = cells_q[int(k[1:])] if k.startswith("q") else v.ljust(14, "_")
            rows_all.append([f"W{wi}", sid, p, k, v, cells])
            for ch in v:
                if ch.isdigit():
                    digit_count[ch] += 1
        # ---- ใบแนว: วาดตัวเลขลงช่องของ template
        img = Image.fromarray(cv2.cvtColor(core.to_std(cv2.imread(TPL[p])), cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(img)

        def put(rect, chars, n):
            x, y, w, h = rect
            cw = w / n
            for i, ch in enumerate(chars):
                if ch in "_.":
                    continue
                cx, cy = x + (i + 0.5) * cw, y + h / 2
                dr.text((cx, cy), ch, font=fD, fill=(25, 60, 150), anchor="mm")

        for qi, r in enumerate(L["rows"]):
            put(r, cells_q[(p - 1) * 30 + qi + 1], 10)
        for name, r in L["header"].items():
            put(r, header[name].ljust(14, "_"), 14)
        # รหัสชุดบนบรรทัดชื่อ (ให้คนเขียนลอกด้วย เพื่อจับคู่รูปกับเฉลย)
        dr.text((250, 145), f"{sid.replace('_', '-')}  หน้า {p}", font=fTag, fill=(190, 40, 40), anchor="lm")
        img.save(f"{OUT}/guides_png/{sid}_p{p}.png")
        writer_pages[wi].append(img)

with open(f"{OUT}/answer_keys_all.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp); w.writerow(["writer", "sheet", "page", "field", "value", "cells"])
    w.writerows(rows_all)

# ---- หน้าคำแนะนำ (ภาษาไทย)
W, H = writer_pages[1][0].size
ins = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(ins)
fH = ImageFont.truetype(FONT_THB, 64); fB = ImageFont.truetype(FONT_TH, 40); fBb = ImageFont.truetype(FONT_THB, 40)
lines = [
    (fH, "วิธีเขียน Dataset รอบ 2 (ตัวเลขผสม)"), (fB, ""),
    (fBb, "1. แต่ละคนรับไฟล์ของตัวเอง (W1 ... W5) = 10 แผ่น"),
    (fB, "    ปริ้นกระดาษเปล่า Template 3 หน้า 1 (ข้อ 1-30) จำนวน 10 แผ่น"),
    (fBb, "2. ลอกตัวเลขจากใบแนวลงช่องเดียวกันทุกช่อง"),
    (fB, "    จำนวนเต็ม: ชิดขวา ติดกับช่องจุดทศนิยม"),
    (fB, "    ทศนิยม: ชิดซ้าย ต่อจากจุดทศนิยม"),
    (fB, "    ช่องที่ใบแนวว่าง ให้เว้นว่าง (ห้ามเขียน 0 เติม)"),
    (fB, "    ช่องรหัสด้านบน: เขียนชิดซ้ายตามใบแนว"),
    (fBb, "3. เขียนรหัสแผ่นและเลขหน้า (เช่น W1-01 หน้า 1) ที่บรรทัดชื่อ"),
    (fBb, "4. ใช้ปากกาสีน้ำเงินหรือดำ เขียนตามลายมือปกติของตัวเอง"),
    (fB, "    ไม่ต้องเขียนให้สวยกว่าปกติ ตัวเลขอยู่ในกรอบ ไม่ทับเส้น"),
    (fBb, "5. ถ่ายรูปทั้งแผ่น ให้เห็นสี่เหลี่ยมดำครบทั้ง 3 มุม"),
    (fB, "    ส่งไฟล์ต้นฉบับจากกล้อง (ห้ามแคปหน้าจอ / ห้ามส่งผ่าน LINE แบบย่อ)"),
    (fB, "    ตั้งชื่อไฟล์: W1_01_p1.jpg, W1_02_p1.jpg, ... W1_10_p1.jpg"),
    (fBb, "6. เขียนคนเดียวทั้ง 10 แผ่น (ห้ามให้คนอื่นช่วยเขียน)"),
    (fB, ""),
    (fBb, "ถ้าเขียนผิด: ขีดฆ่าแล้วแจ้งเลขข้อ หรือเริ่มแผ่นใหม่"),
]
y = 160
for f, t in lines:
    d.text((120, y), t, font=f, fill=(20, 30, 45)); y += 78
ins.save(f"{OUT}/instructions.png")
for wi, pages in writer_pages.items():
    pdf = [ins] + pages
    pdf[0].save(f"{OUT}/guides_W{wi}.pdf", save_all=True, append_images=pdf[1:], resolution=200)

# ---- สรุป
lens = {}
for _, s, p, k, v, c in rows_all:
    if k.startswith("q"):
        ip, _, dp = v.partition(".")
        lens.setdefault(("int", len(ip)), 0); lens[("int", len(ip))] += 1
        lens.setdefault(("dec", len(dp)), 0); lens[("dec", len(dp))] += 1
summary = {"writers": N_WRITERS, "sheets": N_WRITERS * SHEETS_PER_WRITER, "answers": N_WRITERS * SHEETS_PER_WRITER * 30,
           "digits_total": sum(digit_count.values()), "digit_count": digit_count,
           "int_len": {k[1]: v for k, v in sorted(lens.items()) if k[0] == "int"},
           "dec_len": {k[1]: v for k, v in sorted(lens.items()) if k[0] == "dec"}}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
