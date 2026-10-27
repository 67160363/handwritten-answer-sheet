"""เว็บเดโม "ตรวจไว" (Gradio) — ใช้ pipeline เดียวกับ step5_predict.py + out/digitnet_t3.pt
รัน:  python demo_app.py   แล้วเปิด http://<IP เครื่องนี้>:7860 จากมือถือในวง Wi-Fi เดียวกัน
"""
import os
import cv2, numpy as np, pandas as pd
import gradio as gr
import step5_predict as S5
from step5_predict import core, process_sheet, read_cells, yolo_boxes, match, YOLO_CLASSES, CONF_TH

ALIGN_FAIL = "ถ่ายใหม่: ให้เห็นสี่เหลี่ยมดำครบ 3 มุม"
# BGR
GREEN, RED, BLUE, YELLOW, GRAY = (40, 160, 40), (30, 30, 220), (200, 90, 20), (0, 200, 255), (150, 150, 150)
STATUS_COLOR = {"ถูก": GREEN, "ผิด": RED, "อ่านได้": BLUE, "ไม่ตอบ": GRAY}


def read_sheet_detail(path):
    """เหมือน step5.read_sheet ทุกขั้น แต่คืนกรอบแต่ละข้อ + confidence ต่ำสุดไว้วาดผล"""
    aligned, page, L, H, rot, score = process_sheet(path)
    g = core.flatten(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
    boxes = yolo_boxes(aligned)
    header = {}
    for name, r in L["header"].items():
        rect, _ = match(r, boxes, YOLO_CLASSES.index(name))
        header[name] = "".join(c for c, _ in read_cells(core.split_row(g, rect, n=core.HEADER_CELLS)))
    rows = []
    for q, r in enumerate(L["rows"]):
        rect, _ = match(r, boxes, 0)
        cells = core.split_row(g, rect)
        chars = read_cells(cells[:core.DECIMAL_IDX] + cells[core.DECIMAL_IDX + 1:])   # ข้ามช่องจุดทศนิยม
        intp = "".join(c for c, _ in chars[:core.DECIMAL_IDX])
        decp = "".join(c for c, _ in chars[core.DECIMAL_IDX:])
        rows.append({"qn": (page - 1) * 30 + q + 1, "rect": rect,
                     "value": (intp or ("0" if decp else "")) + ("." + decp if decp else ""),
                     "conf": min(p for _, p in chars)})
    return aligned, page, score, header, rows


def parse_key(text, file):
    """เฉลย: บรรทัดละ 'ข้อ,ค่า' (1,12.5 หรือ q1,12.5 หรือ student_id,...) และ/หรือไฟล์ csv แบบ data/eval"""
    key = {}
    if file is not None:
        df = pd.read_csv(file if isinstance(file, str) else file.name, dtype=str)
        for r in df.itertuples():
            key[str(r.field).strip()] = "" if pd.isna(r.value) else str(r.value).strip()
    for line in (text or "").splitlines():
        parts = [s.strip() for s in line.replace("\t", ",").split(",")]
        if len(parts) < 2 or not parts[0] or parts[0].lower() == "field":
            continue
        f = parts[0]
        key[f"q{f}" if f.isdigit() else f] = parts[1]
    return key


def draw(aligned, rows):
    """กรอบสีตามสถานะ + ค่าที่อ่านได้ในขอบขาวที่เติมซ้าย/ขวา (คอลัมน์ซ้ายเขียนทางซ้าย คอลัมน์ขวาเขียนทางขวา)"""
    M, FONT, SC, TH = 280, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3
    img = cv2.copyMakeBorder(aligned, 0, 0, M, M, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    mid = core.STD_W / 2
    for r in rows:
        x, y, w, h = [int(v) for v in r["rect"]]
        x += M
        col = STATUS_COLOR[r["status"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 5)
        if r["low"]:
            cv2.rectangle(img, (x - 8, y - 8), (x + w + 8, y + h + 8), YELLOW, 6)
        txt = r["value"] if r["value"] else "-"
        tw = cv2.getTextSize(txt, FONT, SC, TH)[0][0]
        tx = M - 12 - tw if x - M + w / 2 < mid else core.STD_W + M + 12
        cv2.putText(img, txt, (tx, y + h - 14), FONT, SC, col, TH, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def check(image_path, key_text, key_file):
    empty = pd.DataFrame(columns=["ข้อ", "อ่านได้", "เฉลย", "สถานะ"])
    if not image_path:
        return None, "### อัปโหลดรูปหรือถ่ายจากกล้องก่อน", empty
    try:
        aligned, page, score, header, rows = read_sheet_detail(image_path)
    except Exception:
        return None, f"## ⚠ {ALIGN_FAIL}", empty
    try:
        key = parse_key(key_text, key_file)
    except Exception as e:
        return None, f"## ⚠ อ่านไฟล์เฉลยไม่ได้: {e}", empty
    has_key = any(f"q{r['qn']}" in key for r in rows)
    n_ok = n_key = 0
    for r in rows:
        r["key"] = key.get(f"q{r['qn']}")
        r["low"] = bool(r["value"]) and r["conf"] < CONF_TH
        if not r["value"]:
            r["status"] = "ไม่ตอบ"
        elif r["key"] is None:
            r["status"] = "อ่านได้"
        else:
            r["status"] = "ถูก" if r["value"] == r["key"] else "ผิด"
        if r["key"] is not None:
            n_key += 1
            n_ok += r["value"] == r["key"]
    table = pd.DataFrame([{"ข้อ": r["qn"], "อ่านได้": r["value"] or "-",
                           "เฉลย": "" if r["key"] is None else (r["key"] or "-"),
                           "สถานะ": r["status"] + (" ⚠มั่นใจต่ำ" if r["low"] else "")} for r in rows])
    hdr = " · ".join(f"{n}: **{v or '-'}**" for n, v in
                     zip(["รหัสประจำตัว", "รหัสวิชา", "รหัสสถานศึกษา"], header.values()))
    if has_key:
        top = f"# คะแนน {n_ok} / {n_key}"
    else:
        top = f"# อ่านได้ {sum(bool(r['value']) for r in rows)} / {len(rows)} ข้อ"
    n_low = sum(r["low"] for r in rows)
    info = (f"{top}\n\nหน้า {page} · align score {score:.3f} · {hdr}\n\n"
            + (f"⚠ มั่นใจต่ำ (< {CONF_TH}) {n_low} ข้อ — กรอบเหลือง ควรให้คนตรวจซ้ำ" if n_low else ""))
    return draw(aligned, rows), info, table


LEGEND = ("**สีกรอบ:** 🟩 ถูก · 🟥 ผิด · 🟦 อ่านได้ (ไม่มีเฉลย) · 🟨 มั่นใจต่ำ · ⬜ เทา = ไม่ตอบ")

with gr.Blocks(title="ตรวจไว") as demo:
    gr.Markdown("# ตรวจไว\nตรวจกระดาษคำตอบ Template 3 ด้วย DigitNet-T3 — ถ่ายให้เห็นสี่เหลี่ยมดำครบ 3 มุม")
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(sources=["upload", "webcam"], type="filepath", label="รูปกระดาษคำตอบ")
            key_text = gr.Textbox(lines=6, label="เฉลย (ไม่บังคับ) — บรรทัดละ ข้อ,ค่า",
                                  placeholder="1,714.38\n2,5\n3,12.25")
            key_file = gr.File(file_types=[".csv"], label="หรือไฟล์เฉลย .csv (field,value แบบ data/eval)")
            btn = gr.Button("ตรวจ", variant="primary")
        with gr.Column(scale=2):
            info = gr.Markdown()
            gr.Markdown(LEGEND)
            img_out = gr.Image(label="ผลตรวจ", type="numpy")
            table = gr.Dataframe(label="รายข้อ", interactive=False)
    btn.click(check, [img_in, key_text, key_file], [img_out, info, table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
