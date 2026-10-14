import cv2, json, numpy as np

STD_W, STD_H = 1654, 2339          # ขนาดมาตรฐานหลังจัดภาพ (A4 @200dpi)
N_CELLS = 10                        # ช่องต่อแถวคำตอบ
DECIMAL_IDX = 6                     # ช่องที่ 7 (index 6) = จุดทศนิยม (พิมพ์ไว้แล้ว)
HEADER_CELLS = 14                   # ช่องต่อแถวรหัส

def to_std(img):
    return cv2.resize(img, (STD_W, STD_H), interpolation=cv2.INTER_AREA)

def flatten(gray):
    """แก้แสงไม่สม่ำเสมอ: หารด้วยภาพพื้นหลัง (max-filter) -> พื้นกระดาษขาวเท่ากันทั้งภาพ"""
    h, w = gray.shape
    s = 400 / max(h, w)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    k = max(3, int(min(small.shape) / 10) | 1)
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (k, k), 0)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.clip(gray.astype(np.float32) / (bg + 1) * 255, 0, 255).astype(np.uint8)

def binarize(gray):
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 51, 15)

# ---------------------------------------------------------------- markers
def find_markers(img):
    """หาจุดอ้างอิง 4 จุด: สี่เหลี่ยมดำล่างซ้าย, ล่างขวา, กล่องเลข 3 ขวาบน, ขีดดำแถวแรกซ้าย
    คืนค่า dict {name: (cx, cy)} หรือ None"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = gray.shape
    th = (flatten(gray) < 110).astype(np.uint8) * 255
    k = max(2, round(5 * max(H, W) / STD_H))          # kernel ตามขนาดภาพ (ภาพเล็กใช้ kernel เล็ก กันขีดติดกรอบ)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = H * W
    squares, dashes = [], []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        area = cv2.countNonZero(th[y:y + h, x:x + w])     # นับพิกเซลจริง (แม่นกว่า contourArea ตอนภาพเล็ก)
        ar = w / h
        fill = area / (w * h)
        # สี่เหลี่ยมทึบใหญ่ ~0.25% ของหน้า
        if 0.7 < ar < 1.4 and fill > 0.8 and 0.0008 < area / page_area < 0.01:
            squares.append((x + w / 2, y + h / 2, area))
        # ขีด timing mark (ยาว:สูง ≈ 4:1)
        elif 2.0 < ar < 7 and fill > 0.75 and 0.00005 < area / page_area < 0.001:
            dashes.append((x + w / 2, y + h / 2))
    if len(squares) < 3:
        return None
    # ใช้ 3 สี่เหลี่ยมที่ใหญ่ที่สุด
    squares = sorted(squares, key=lambda s: -s[2])[:3]
    tr = min(squares, key=lambda s: s[1] - s[0])          # บนสุด-ขวาสุด
    rest = [s for s in squares if s is not tr]
    bl, br = sorted(rest, key=lambda s: s[0])
    # ขีดแถวแรกฝั่งซ้าย = ขีดที่ x < กลางหน้า และ y น้อยสุด
    left = [d for d in dashes if d[0] < (bl[0] + br[0]) / 2 and d[1] < bl[1]]
    tl = min(left, key=lambda d: d[1]) if left else None   # ไม่เจอขีด -> ใช้ affine 3 จุด
    return {"tl": tl[:2] if tl else None, "tr": tr[:2], "bl": bl[:2], "br": br[:2]}

def marker_array(m):
    return np.float32([m["tl"], m["tr"], m["br"], m["bl"]])

# ---------------------------------------------------------------- layout (วัดจาก template เปล่า)
def _rects(th, min_w, max_w, ar_lo, ar_hi):
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if min_w <= w <= max_w and ar_lo <= w / h <= ar_hi:
            out.append([x, y, w, h])
    return out

def build_layout(template_img):
    """วัดพิกัดกล่องทั้งหมดจาก template เปล่า (ทำครั้งเดียว) -> dict"""
    img = to_std(template_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ดึงเฉพาะเส้นแนวนอน/แนวตั้งยาว ๆ เพื่อหากรอบตาราง
    hl = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    vl = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    grid = cv2.dilate(hl | vl, np.ones((3, 3), np.uint8))

    # แถวคำตอบ: กว้าง ~ 1/3 หน้า, อัตราส่วน ~9.5
    rows = _rects(grid, STD_W * 0.25, STD_W * 0.45, 7, 13)
    rows = sorted(rows, key=lambda r: (r[0] > STD_W / 2, r[1]))   # คอลัมน์ซ้ายก่อน แล้วบนลงล่าง
    # กล่องรหัส 3 แถว (เฉพาะส่วนช่องกรอก ไม่รวมหัวข้อ): กว้าง ~ 0.48 หน้า อัตราส่วน ~ 13.8
    # กรอบรวม (หัวข้อ+ช่อง) อัตราส่วน ~7 -> เอาครึ่งล่างเป็นแถวช่องกรอก
    hdr = _rects(grid, STD_W * 0.4, STD_W * 0.55, 5, 9.5)
    hdr = [[x, y + h // 2, w, h - h // 2] for x, y, w, h in sorted(hdr, key=lambda r: r[1])]
    # ช่องฝนวงกลม (ชุดคำถาม/หน้า) — วงกลมเล็ก
    cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 25 <= w <= 45 and 0.8 < w / h < 1.25 and y < STD_H * 0.25 and x < STD_W * 0.4:
            per = cv2.arcLength(c, True)
            if per > 0 and 4 * np.pi * cv2.contourArea(c) / per ** 2 > 0.6:
                circles.append([x, y, w, h])
    circles = _dedup(circles)
    circles = sorted(circles, key=lambda r: r[0])
    layout = {
        "size": [STD_W, STD_H],
        "markers": {k: list(map(float, v)) for k, v in find_markers(img).items()},
        "rows": rows,
        "header": dict(zip(["student_id", "subject_id", "school_id"], hdr)),
        "quiz_set": circles[:3],
        "page_no": circles[3:6],
    }
    return layout

def _dedup(rs, tol=8):
    out = []
    for r in rs:
        if all(abs(r[0] - o[0]) > tol or abs(r[1] - o[1]) > tol for o in out):
            out.append(r)
    return out

# ---------------------------------------------------------------- align
def find_page(img):
    """(ทางเลือก) หาขอบกระดาษ 4 มุม แล้วตัดพื้นหลังออก ถ้าไม่เจอคืนภาพเดิม"""
    ratio = 1000 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=ratio, fy=ratio)
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(g, 50, 150), np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < 0.3 * small.shape[0] * small.shape[1]:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            pts = _order(approx.reshape(4, 2) / ratio)
            M = cv2.getPerspectiveTransform(pts, np.float32([[0, 0], [STD_W, 0], [STD_W, STD_H], [0, STD_H]]))
            return cv2.warpPerspective(img, M, (STD_W, STD_H)), M
    return img, None

def _order(p):
    p = np.float32(p)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.float32([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]])

def _align_once(img, layout):
    pre, M0 = find_page(img)
    m = find_markers(pre)
    if m is None and M0 is not None:          # ตัดขอบกระดาษแล้วหาไม่เจอ -> ลองบนภาพเดิม
        pre, M0 = img, None
        m = find_markers(pre)
    if m is None:
        return None, None
    if m["tl"] is not None:
        H = cv2.getPerspectiveTransform(marker_array(m), marker_array(layout["markers"]))
    else:
        k = ["tr", "br", "bl"]
        A = cv2.getAffineTransform(np.float32([m[i] for i in k]), np.float32([layout["markers"][i] for i in k]))
        H = np.vstack([A, [0, 0, 1]])
    aligned = cv2.warpPerspective(pre, H, (STD_W, STD_H), borderValue=(255, 255, 255))
    return aligned, (H @ M0 if M0 is not None else H)

def match_score(aligned, ref_gray):
    """ความเหมือนกับ template (ใช้ตรวจว่าจัดภาพถูกต้อง/ไม่กลับหัว)"""
    a = cv2.resize(flatten(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)), (200, 283)).astype(np.float32)
    b = cv2.resize(ref_gray, (200, 283)).astype(np.float32)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

ROTS = [None, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]

def align(img, layout, ref_gray, min_score=0.4):
    """จัดภาพถ่ายให้ตรง template ลองหมุน 0/180/90/270 แล้วเลือกที่เหมือน template ที่สุด
    คืน (aligned_img, H, rot, score) — H แมปจากภาพที่หมุนแล้ว -> ภาพมาตรฐาน"""
    best = (None, None, None, -1)
    for rot in ROTS:
        im = img if rot is None else cv2.rotate(img, rot)
        al, H = _align_once(im, layout)
        if al is None:
            continue
        sc = match_score(al, ref_gray)
        if sc > best[3]:
            best = (al, H, rot, sc)
        if sc > 0.8:
            break
    if best[0] is None or best[3] < min_score:
        raise RuntimeError(f"จัดภาพไม่สำเร็จ (score={best[3]:.2f}) — ตรวจว่าเห็นสี่เหลี่ยมดำครบ 3 มุม")
    return best

# ---------------------------------------------------------------- cells
def refine_rect(gray, rect, search=15):
    """ขยับขอบกรอบทั้ง 4 ด้านให้ทับเส้นขอบจริงของกล่อง (หาแยกซ้าย/ขวา/บน/ล่าง)
    แก้ความคลาดเคลื่อนจากกระดาษโค้ง ภาพเล็ก หรือขนาดพิมพ์ต่างกันเล็กน้อย"""
    x, y, w, h = map(int, rect)
    s = search
    H, W = gray.shape
    if x - s < 0 or y - s < 0 or x + w + s >= W or y + h + s >= H:
        return [x, y, w, h]
    dark = (gray[y - s:y + h + s + 1, x - s:x + w + s + 1] < 150).astype(np.float32)
    k = np.ones(3) / 3

    def edge(profile, nominal):
        seg = np.convolve(profile, k, 'same')[nominal - s:nominal + s + 1]
        d = int(np.argmax(seg - 0.002 * np.abs(np.arange(-s, s + 1))))   # เท่ากัน -> เลือกที่ใกล้เดิม
        return d - s if seg.max() > 0.5 else 0                            # ไม่เจอเส้นชัด -> ไม่ขยับ

    rp = dark[:, s + 5:s + w - 5].mean(1)                 # เส้นแนวนอน (บน/ล่าง)
    t, bt = edge(rp, s), edge(rp, s + h)
    cp = dark[s + t + 5:s + h + bt - 5, :].mean(0)        # เส้นแนวตั้ง (ซ้าย/ขวา) ไม่นับเส้นบนล่าง
    l, r = edge(cp, s), edge(cp, s + w)
    return [x + l, y + t, w + r - l, h + bt - t]

def split_row(aligned, rect, n=N_CELLS, pad=0.14, refine=True):
    """แบ่งกรอบแถวเป็น n ช่องเท่ากัน ตัดขอบเส้นออก pad% -> list ของภาพ grayscale"""
    x, y, w, h = rect
    # ส่งภาพ gray ที่ flatten แล้วมาจะเร็วกว่า (ทำครั้งเดียวต่อหน้า)
    gray = flatten(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)) if aligned.ndim == 3 else aligned
    if refine:
        x, y, w, h = refine_rect(gray, rect)
    cw = w / n
    cells = []
    for i in range(n):
        x0, x1 = x + i * cw, x + (i + 1) * cw
        px, py = cw * pad, h * pad
        cells.append(gray[int(y + py):int(y + h - py), int(x0 + px):int(x1 - px)])
    return cells

INK_LEVEL = 185   # ค่าความสว่าง (หลัง flatten) ที่ถือว่าเป็นหมึกสำหรับ Character Check — กระดาษ ~255

DIGIT_LEVEL = 200  # threshold สำหรับตัดรูปตัวเลข (หลวมกว่า เก็บเส้นจาง)

def ink_mask(cell, level=None):
    m = (cell < (level or INK_LEVEL)).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))   # ลบจุด noise

def remove_border_lines(th):
    """ลบเศษเส้นตารางที่ติดขอบช่อง (เส้นยาวบาง แนวตั้ง/แนวนอน ที่แตะขอบ)"""
    h, w = th.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(th)
    for i in range(1, n):
        x, y, bw, bh = st[i, :4]
        vline = bh > 0.5 * h and bw < 0.15 * w and (x == 0 or x + bw >= w)          # ต้องแตะขอบจริง (กันลบเลข 1)
        hline = bw > 0.5 * w and bh < 0.15 * h and (y <= 0.15 * h or y + bh >= 0.85 * h)
        if vline or hline:
            th[lab == i] = 0
    return th

def ink_ratio(cell):
    """สัดส่วนหมึกในช่อง (Character Check: ว่างหรือไม่) ไม่นับเศษเส้นตาราง"""
    return float(remove_border_lines(ink_mask(cell)).mean() / 255)

def bubble_filled(aligned, circles):
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    vals = [255 - gray[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4].mean() for x, y, w, h in circles]
    if max(vals) < 80:                        # ไม่มีวงไหนถูกฝน
        return 0, vals
    return int(np.argmax(vals)) + 1, vals

def prep_cell(cell, size=32):
    """ทำให้ช่องเป็นภาพขาวดำ 32x32 ตัวเลขอยู่กลาง (แบบ MNIST)"""
    th = ink_mask(cell, level=DIGIT_LEVEL)          # threshold หลวมกว่า -> เส้นปากกาจาง ๆ ไม่ขาด
    th = remove_border_lines(th)
    # เก็บเฉพาะ component ที่ใหญ่พอ (ตัดเศษเส้นตาราง/จุด)
    n, lab, st, _ = cv2.connectedComponentsWithStats(th)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.15 * st[1:, cv2.CC_STAT_AREA].max()] if n > 1 else []
    th = np.isin(lab, keep).astype(np.uint8) * 255
    ys, xs = np.nonzero(th)
    if len(xs) < 15:
        return np.zeros((size, size), np.uint8)
    d = th[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = max(d.shape) + 8
    canvas = np.zeros((s, s), np.uint8)
    oy, ox = (s - d.shape[0]) // 2, (s - d.shape[1]) // 2
    canvas[oy:oy + d.shape[0], ox:ox + d.shape[1]] = d
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
