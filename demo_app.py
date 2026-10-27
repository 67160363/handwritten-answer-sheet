"""เว็บเดโม "ตรวจไว" (Gradio) — ใช้ pipeline เดียวกับ step5_predict.py + out/digitnet_t3.pt
รัน:  python demo_app.py   แล้วเปิด http://<IP เครื่องนี้>:7860 จากมือถือในวง Wi-Fi เดียวกัน
"""
import os, tempfile
import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F
import gradio as gr
import step5_predict as S5
from step5_predict import core, read_cells, yolo_boxes, match, YOLO_CLASSES, CONF_TH
from common import load_layouts, imread, template_path, INK_TH, CLASSES
from digitnet import plain
import t3ink

DEMO_MIN_SCORE = 0.5     # เดโมยอมรับรูปที่จัดได้คะแนนต่ำกว่า step5 (0.7) เช่นรูปแสงน้อยมีเงา
ALIGN_FAIL = "ถ่ายใหม่: วางกระดาษบนโต๊ะ ถ่ายจากด้านบน ให้เห็นสี่เหลี่ยมดำครบ 3 มุม"
# BGR
GREEN, RED, BLUE, YELLOW, GRAY = (40, 160, 40), (30, 30, 220), (200, 90, 20), (0, 200, 255), (150, 150, 150)
STATUS_COLOR = {"ถูก": GREEN, "ผิด": RED, "อ่านได้": BLUE, "ไม่ตอบ": GRAY}


_LMASK = {}


def lmask(page):
    """มาสก์สิ่งพิมพ์ของ template หน้านั้น (ใช้ตอนตัดช่องตามหมึก) — คำนวณครั้งเดียวต่อหน้า"""
    if page not in _LMASK:
        _LMASK[page] = t3ink.line_mask(imread(template_path(page)))
    return _LMASK[page]


@torch.no_grad()
def read_masks(masks):
    """ช่องจากการตัดตามหมึก (มาสก์ตัวเลข / None = ว่าง) -> [(ตัวอักษร, confidence)] ผ่าน to32 + CNN ตัวเดียวกัน"""
    out = [("", 1.0) if m is None else None for m in masks]
    idx = [k for k, m in enumerate(masks) if m is not None]
    if idx:
        pr = F.softmax(S5.model(torch.stack([plain(t3ink.to32(masks[k])) for k in idx]).to(S5.dev)), 1).cpu()
        for k, q in zip(idx, pr):
            j = int(q.argmax())
            out[k] = ("" if CLASSES[j] == "blank" else CLASSES[j], float(q[j]))
    return out


def read_sheet_detail(path, hybrid=True):
    """เหมือน step5.read_sheet แต่ (1) เกณฑ์ align หลวมกว่า (DEMO_MIN_SCORE) (2) hybrid=True: แถวที่รูปแบบหมึก
    ผิดปกติ (ทับเส้น/ลอย/ล้น) ตัดช่องตามหมึกด้วย t3ink แถวปกติใช้วิธีเดิม ; คืนกรอบแต่ละข้อ + confidence ต่ำสุด"""
    LAYOUT, REF = load_layouts()
    aligned, H, rot, score = core.align(imread(path), LAYOUT[1], [REF[p] for p in (1, 2, 3)], min_score=DEMO_MIN_SCORE)
    page = core.bubble_filled(aligned, LAYOUT[1]["page_no"])[0] or 1
    L = LAYOUT[page]
    g = core.flatten(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY))
    boxes = yolo_boxes(aligned)
    header = {}
    for name, r in L["header"].items():
        rect, _ = match(r, boxes, YOLO_CLASSES.index(name))
        header[name] = "".join(c for c, _ in read_cells(core.split_row(g, rect, n=core.HEADER_CELLS)))
    rects = [match(r, boxes, 0)[0] for r in L["rows"]]
    hy = t3ink.page_cells_hybrid(g, dict(L, rows=rects), lmask(page), ink_th=INK_TH) if hybrid else {}
    rows = []
    for q, rect in enumerate(rects):
        if hybrid and hy[q][0] == "ink":
            masks = hy[q][1]
            chars = read_masks(masks[:core.DECIMAL_IDX] + masks[core.DECIMAL_IDX + 1:])
        else:
            cells = core.split_row(g, rect)
            chars = read_cells(cells[:core.DECIMAL_IDX] + cells[core.DECIMAL_IDX + 1:])   # ข้ามช่องจุดทศนิยม
        intp = "".join(c for c, _ in chars[:core.DECIMAL_IDX])
        decp = "".join(c for c, _ in chars[core.DECIMAL_IDX:])
        rows.append({"qn": (page - 1) * 30 + q + 1, "rect": rect, "method": hy[q][0] if hybrid else "grid",
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


def grade(image_path, key_text, key_file):
    """อ่านแผ่น + เทียบเฉลย -> (ภาพผล, page, score, header, rows, has_key) ; จัดภาพไม่ได้ -> RuntimeError(ALIGN_FAIL)"""
    try:
        aligned, page, score, header, rows = read_sheet_detail(image_path)
    except Exception:
        raise RuntimeError(ALIGN_FAIL)
    key = parse_key(key_text, key_file)
    has_key = any(f"q{r['qn']}" in key for r in rows)
    for r in rows:
        r["key"] = key.get(f"q{r['qn']}")
        r["low"] = bool(r["value"]) and r["conf"] < CONF_TH
        if not r["value"]:
            r["status"] = "ถูก" if r["key"] == "" else "ไม่ตอบ"     # เฉลยให้เว้นว่าง + เว้นว่างจริง = ถูก (นับเหมือน v1)
        elif r["key"] is None:
            r["status"] = "อ่านได้"
        else:
            r["status"] = "ถูก" if r["value"] == r["key"] else "ผิด"
    return draw(aligned, rows), page, score, header, rows, has_key


# ---------------------------------------------------------------- หน้าเว็บ
NAVY, ORANGE, BLUE_W, BG = "#172a3a", "#f2a65a", "#2f6690", "#f6f4ee"
BADGE = {"ถูก": "ok", "ผิด": "bad", "อ่านได้": "read", "ไม่ตอบ": "none"}     # สถานะ -> คลาส css
HEADER_NAMES = ["รหัสประจำตัว", "รหัสวิชา", "รหัสสถานศึกษา"]


def score_card(page, header, rows, has_key):
    n_ok = sum(r["status"] == "ถูก" for r in rows)
    n_key = sum(r["key"] is not None for r in rows)
    n_read = sum(bool(r["value"]) for r in rows)
    big = f"{n_ok} <small>/ {n_key}</small>" if has_key else f"อ่านได้ {n_read} <small>ข้อ</small>"
    chips = [("ok", "ถูก", n_ok), ("bad", "ผิด", sum(r["status"] == "ผิด" for r in rows))] if has_key \
        else [("read", "อ่านได้", n_read)]
    chips += [("low", "ให้ครูดู", sum(r["low"] for r in rows)), ("none", "ไม่ตอบ", sum(r["status"] == "ไม่ตอบ" for r in rows))]
    chip_html = "".join(f"<span class='tv-chip {c}'>{t} <b>{n}</b></span>" for c, t, n in chips)
    ids = "".join(f"<span><b>{n}</b>{v or '-'}</span>" for n, v in zip(HEADER_NAMES, header.values()))
    return (f"<div class='tv-score'><div class='tv-score-label'>{'คะแนน' if has_key else 'ผลการอ่าน'} · หน้า {page}</div>"
            f"<div class='tv-score-big'>{big}</div><div class='tv-chips'>{chip_html}</div><div class='tv-ids'>{ids}</div></div>")


def item_list(rows):
    def item(r):
        key = f"<span class='tv-key'>เฉลย {r['key'] or '-'}</span>" if r["status"] == "ผิด" else ""
        low = "<span class='tv-badge low'>ให้ครูดู</span>" if r["low"] else ""
        return (f"<div class='tv-item'><span class='tv-q'>ข้อ {r['qn']}</span>"
                f"<span class='tv-ans'><span class='tv-val'>{r['value'] or '–'}</span>{key}</span><span class='tv-badges'>{low}<span class='tv-badge {BADGE[r['status']]}'>{r['status']}</span></span></div>")
    return f"<div class='tv-list'>{''.join(item(r) for r in rows)}</div>"


def to_excel(rows, header):
    path = os.path.join(tempfile.mkdtemp(), "ตรวจไว_ผลตรวจ.xlsx")
    items = pd.DataFrame([{"ข้อ": r["qn"], "อ่านได้": r["value"], "เฉลย": "" if r["key"] is None else r["key"],
                           "สถานะ": r["status"], "ให้ครูดู": "ใช่" if r["low"] else "",
                           "confidence": round(r["conf"], 3)} for r in rows])
    with pd.ExcelWriter(path) as xw:
        items.to_excel(xw, sheet_name="รายข้อ", index=False)
        pd.DataFrame({"ฟิลด์": HEADER_NAMES, "ค่า": list(header.values())}).to_excel(xw, sheet_name="ข้อมูลผู้สอบ", index=False)
    return path


PLACEHOLDER = "<div class='tv-score empty'><div class='tv-score-label'>ผลตรวจจะแสดงที่นี่</div></div>"
LOADING = "<div class='tv-score empty'><div class='tv-spinner'></div><div class='tv-score-label'>กำลังอ่านกระดาษ…</div></div>"
LEGEND = "<div class='tv-legend'>" + "".join(
    f"<span><i style='background:{c}'></i>{t}</span>" for c, t in
    [("#28a028", "ถูก"), ("#dc1e1e", "ผิด"), ("#145ac8", "อ่านได้"), ("#ffc800", "ให้ครูดู"), ("#969696", "ไม่ตอบ")]) + "</div>"


def warn(title, body=""):
    return f"<div class='tv-warn'><b>⚠ {title}</b>{body}</div>"


def loading():
    return LOADING, gr.update(visible=False), None, "", gr.update(value=None, visible=False)


def check(image_path, key_text, key_file):
    fail = lambda html: (html, gr.update(visible=False), None, "", gr.update(value=None, visible=False))
    if not image_path:
        return fail(warn("ยังไม่มีรูป", ": อัปโหลดหรือถ่ายรูปกระดาษคำตอบก่อน"))
    try:
        img, page, score, header, rows, has_key = grade(image_path, key_text, key_file)
    except RuntimeError:
        return fail(warn("ถ่ายใหม่", ALIGN_FAIL[len("ถ่ายใหม่"):]))
    except Exception as e:
        return fail(warn("อ่านเฉลยไม่ได้", f": {e}"))
    return (score_card(page, header, rows, has_key), gr.update(visible=True), img, item_list(rows),
            gr.update(value=to_excel(rows, header), visible=True))


def step(n, title, sub=""):
    return gr.HTML(f"<div class='tv-step'><span class='tv-num'>{n}</span><div><div class='tv-step-title'>{title}</div>"
                   f"<div class='tv-step-sub'>{sub}</div></div></div>")


HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Prompt:wght@500;600;700'
        '&family=Sarabun:wght@400;500;700&display=swap" rel="stylesheet">')

CSS = """
:root { --tv-navy:#172a3a; --tv-orange:#f2a65a; --tv-blue:#2f6690; --tv-bg:#f6f4ee; }
body, gradio-app, .gradio-container { background:var(--tv-bg) !important; color:var(--tv-navy); font-family:'Sarabun',sans-serif; }
.gradio-container { max-width:1240px !important; width:100% !important; margin:0 auto !important; padding:0 16px 32px !important;
                    overflow:visible !important; }
footer { display:none !important; }
.tv-step-title, .tv-score-big, .tv-logo, button { font-family:'Prompt',sans-serif !important; }
.tv-hero { background:var(--tv-navy); color:#fff; width:100vw; margin:0 0 8px calc(50% - 50vw); padding:26px 16px; box-sizing:border-box; }
body { overflow-x:hidden; }
#tv-hero-wrap, #tv-hero-wrap > div { overflow:visible !important; padding:0 !important; }
.tv-hero *, .tv-score:not(.empty) * { color:#fff !important; }
.tv-hero .tv-logo span { color:var(--tv-orange) !important; }
.tv-hero .tv-mark { color:var(--tv-navy) !important; }
.tv-score .tv-chip.ok, .tv-score .tv-chip.ok * { color:#2e6b45 !important; }
.tv-score .tv-chip.bad, .tv-score .tv-chip.bad * { color:#a44e16 !important; }
.tv-score .tv-chip.low, .tv-score .tv-chip.low * { color:#7a5a00 !important; }
.tv-score .tv-chip.none, .tv-score .tv-chip.none * { color:#5f6368 !important; }
.tv-score .tv-chip.read, .tv-score .tv-chip.read * { color:var(--tv-blue) !important; }
.tv-hero-inner { max-width:1240px; margin:0 auto; display:flex; align-items:center; gap:16px; }
.tv-mark { flex:none; width:56px; height:56px; border-radius:16px; background:var(--tv-orange); color:var(--tv-navy);
           display:grid; place-items:center; font:700 32px 'Prompt'; }
.tv-logo { font-size:42px; font-weight:700; line-height:1; }
.tv-logo span { color:var(--tv-orange); }
.tv-slogan { font-size:18px; opacity:.85; margin-top:6px; }
.tv-card { background:#fff !important; border-radius:16px !important; box-shadow:0 2px 14px rgba(23,42,58,.08) !important;
           border:1px solid rgba(23,42,58,.06) !important; padding:16px !important; }
.tv-step { display:flex; align-items:center; gap:12px; }
.tv-num { flex:none; width:36px; height:36px; border-radius:50%; background:var(--tv-navy); color:#fff; display:grid;
          place-items:center; font:600 18px 'Prompt'; }
.tv-step-title { font-size:19px; font-weight:600; color:var(--tv-navy); }
.tv-step-sub { font-size:14px; color:#5b6670; }
#tv-go { background:var(--tv-orange) !important; color:var(--tv-navy) !important; border:none !important; font-size:22px !important;
         font-weight:700 !important; padding:16px !important; border-radius:16px !important; box-shadow:0 4px 14px rgba(242,166,90,.45) !important; }
#tv-go:hover { filter:brightness(1.05); }
#tv-dl { background:var(--tv-blue) !important; color:#fff !important; border:none !important; border-radius:16px !important;
         font-size:17px !important; padding:12px !important; }
.tv-score { background:var(--tv-navy); color:#fff; border-radius:16px; padding:22px 24px; box-shadow:0 4px 18px rgba(23,42,58,.18); }
.tv-score.empty { background:#fff; color:var(--tv-navy); text-align:center; padding:40px 24px;
                  border:2px dashed rgba(23,42,58,.15); box-shadow:none; }
.tv-score-label { font-size:16px; opacity:.85; }
.tv-score-big { font-size:64px; font-weight:700; line-height:1.1; margin:4px 0 12px; }
.tv-score-big small { font-size:30px; opacity:.75; }
.tv-chips { display:flex; flex-wrap:wrap; gap:8px; }
.tv-chip { border-radius:999px; padding:5px 14px; font-size:15px; }
.tv-chip.ok, .tv-badge.ok { color:#2e6b45 !important; background:#e3f1e7; }
.tv-chip.bad, .tv-badge.bad { color:#a44e16 !important; background:#fbe9dc; }
.tv-chip.low, .tv-badge.low { color:#7a5a00 !important; background:#fdf1c7; }
.tv-chip.none, .tv-badge.none { color:#5f6368 !important; background:#eceff1; }
.tv-chip.read, .tv-badge.read { color:var(--tv-blue) !important; background:#e2edf6; }
.tv-ids { display:flex; flex-wrap:wrap; gap:4px 18px; margin-top:14px; font-size:14px; opacity:.9; }
.tv-ids b { font-weight:400; opacity:.65; margin-right:6px; }
.tv-legend { display:flex; flex-wrap:wrap; gap:6px 16px; font-size:14px; margin:2px 2px 6px; color:var(--tv-navy); }
.tv-legend i { display:inline-block; width:14px; height:14px; border-radius:4px; margin-right:6px; vertical-align:-2px; }
.tv-list { display:grid; grid-template-columns:1fr 1fr; grid-template-rows:repeat(15, auto); grid-auto-flow:column; gap:8px; }
.tv-item { display:flex; align-items:center; gap:10px; background:#fff; border-radius:12px; padding:9px 12px;
           box-shadow:0 1px 6px rgba(23,42,58,.07); color:var(--tv-navy); }
.tv-q { font:500 14px 'Prompt'; color:#5b6670 !important; min-width:44px; flex:none; }
.tv-ans { display:flex; flex-direction:column; min-width:0; }
.tv-val { font:700 24px/1.15 'Prompt'; color:#172a3a !important; letter-spacing:.5px; }
.tv-key { font-size:14px; font-weight:500; color:#a44e16 !important; white-space:nowrap; }
.tv-badges { margin-left:auto; display:flex; gap:4px; }
.tv-badge { border-radius:999px; padding:2px 10px; font-size:13px; font-weight:600; white-space:nowrap; }
.tv-warn { background:#fdebd8; border:1px solid var(--tv-orange); color:#7a3e0e; border-radius:16px; padding:20px 22px; font-size:17px; }
.tv-warn b { font-family:'Prompt'; font-size:19px; }
.tv-spinner { width:36px; height:36px; margin:0 auto 10px; border-radius:50%; border:4px solid #e6e1d6;
              border-top-color:var(--tv-orange); animation:tvspin .8s linear infinite; }
@keyframes tvspin { to { transform:rotate(360deg); } }
@media (max-width: 820px) {
  #tv-main { flex-direction:column !important; }
  #tv-main > div { width:100% !important; min-width:0 !important; }
  .tv-list { grid-template-columns:1fr; grid-template-rows:none; grid-auto-flow:row; }
  .tv-logo { font-size:34px; } .tv-mark { width:46px; height:46px; font-size:26px; }
  .tv-slogan { font-size:16px; } .tv-score-big { font-size:52px; }
  #tv-go, #tv-dl { width:100% !important; }
}
"""

THEME = gr.themes.Soft(primary_hue=gr.themes.colors.orange, secondary_hue=gr.themes.colors.blue,
                       radius_size=gr.themes.sizes.radius_lg,
                       font=[gr.themes.GoogleFont("Sarabun"), "sans-serif"]).set(
    body_background_fill=BG, body_text_color=NAVY, block_radius="16px", block_shadow="0 2px 14px rgba(23,42,58,.08)",
    button_primary_background_fill=ORANGE, button_primary_text_color=NAVY)

with gr.Blocks(title="ตรวจไว — ตรวจกระดาษคำตอบ") as demo:
    gr.HTML("<div class='tv-hero'><div class='tv-hero-inner'><div class='tv-mark'>✓</div><div>"
            "<div class='tv-logo'>ตรวจ<span>ไว</span></div>"
            "<div class='tv-slogan'>ถ่ายรูปกระดาษคำตอบ ได้คะแนนทันที</div></div></div></div>", elem_id="tv-hero-wrap")
    with gr.Row(elem_id="tv-main", equal_height=False):
        with gr.Column(scale=5, min_width=320):
            with gr.Column(elem_classes="tv-card"):
                step(1, "ใส่เฉลย", "ข้ามได้ ถ้าแค่อยากให้อ่านค่า")
                with gr.Accordion("มีเฉลยไหม (ไม่บังคับ)", open=False):
                    key_text = gr.Textbox(lines=5, label="พิมพ์เฉลย บรรทัดละ ข้อ,ค่า",
                                          placeholder="1,714.38\n2,615.51\n3,5.429")
                    key_file = gr.File(file_types=[".csv"], label="หรืออัปโหลด .csv (field,value)", height=110)
            with gr.Column(elem_classes="tv-card"):
                step(2, "ถ่ายรูปกระดาษ", "วางบนโต๊ะ ถ่ายจากด้านบน ให้เห็นสี่เหลี่ยมดำครบ 3 มุม")
                img_in = gr.Image(sources=["upload", "webcam"], type="filepath", show_label=False, height=320)
            with gr.Column(elem_classes="tv-card"):
                step(3, "ตรวจ")
                btn = gr.Button("ตรวจเลย", elem_id="tv-go", size="lg")
        with gr.Column(scale=7, min_width=320):
            score_out = gr.HTML(PLACEHOLDER)
            with gr.Column(elem_classes="tv-card", visible=False) as res_card:
                gr.HTML(LEGEND)
                img_out = gr.Image(type="numpy", show_label=False, interactive=False)
            list_out = gr.HTML()
            dl = gr.DownloadButton("ดาวน์โหลดผล (Excel)", elem_id="tv-dl", visible=False)
    outs = [score_out, res_card, img_out, list_out, dl]
    btn.click(loading, None, outs, queue=False, show_progress="hidden").then(
        check, [img_in, key_text, key_file], outs, show_progress="hidden")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=THEME, css=CSS, head=HEAD, footer_links=[])
