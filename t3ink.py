"""ตัดช่องตาม "หมึกจริง" แทนกรอบตายตัว — แก้เขียนทับเส้นข้าง / ลอยทับเส้นบน / ตัวใหญ่ล้น
1) ลบเส้นตาราง: ใช้ template เปล่าที่จัดแล้ว -> ตำแหน่งเส้นพิมพ์ (ขยาย 3px) ลบออกจากหมึก
2) หาก้อนหมึก (connected components) ในแถบรอบแถว (เผื่อขึ้นบน 0.9 เท่าความสูงช่อง)
3) ก้อนหมึกแต่ละก้อน -> ไปแถวที่จุดศูนย์กลางใกล้ที่สุด -> ไปช่องตามจุดศูนย์กลางแนวนอน
4) ตัดรูปตัวเลขจากกรอบของก้อนหมึก (ไม่ใช่กรอบช่อง) -> 32x32 แบบ MNIST
ใช้แทน split_row + ink_ratio ได้: row_cells_ink() คืน list 10 ช่อง (None = ว่าง)
"""
import cv2, numpy as np
import t3core as core

INK = 190          # หลัง flatten: หมึก < 200 (หลวม เก็บเส้นปากกาจาง)
MIN_AREA = 120     # พิกเซลหมึกน้อยกว่านี้ = จุด noise / เงา (ตัวเลขจริง ~200-600)


def line_mask(layout_template_img):
    """มาสก์สิ่งพิมพ์ที่ "ไม่ใช่เส้นตาราง" ของ template (พื้นเทาช่องจุด, จุด, เลขข้อ, ขีดดำ) ขยายเล็กน้อย
    เส้นตารางไม่ใส่ในมาสก์ เพราะกระดาษโค้งทำให้เส้นในรูปเลื่อน -> หาเส้นจากรูปจริงแทน (_ink)"""
    g = cv2.cvtColor(core.to_std(layout_template_img), cv2.COLOR_BGR2GRAY)
    m = (g < 235).astype(np.uint8) * 255
    dark = (g < 150).astype(np.uint8) * 255            # เส้นพิมพ์เข้ม (พื้นเทาช่องจุด ~200 ไม่นับ)
    hl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    vl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)))
    m[cv2.dilate(hl | vl, np.ones((3, 3), np.uint8)) > 0] = 0
    return cv2.dilate(m, np.ones((7, 7), np.uint8))


def grid_lines(binary, cell_h=57):
    """หาเส้นตารางจากรูปจริง: เส้นแนวนอนยาว + เส้นแนวตั้งที่สูงเกือบเต็มช่อง และบาง"""
    hl = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1)))
    vl = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(cell_h * 0.8))))
    n, lab, st, _ = cv2.connectedComponentsWithStats(vl)
    for k in range(1, n):                      # เลข 1 เขียนยาว ๆ: หนากว่าเส้นพิมพ์ -> ไม่ลบ
        if st[k, cv2.CC_STAT_WIDTH] > 6:
            vl[lab == k] = 0
    return hl | vl


def _ink(gray, lmask):
    m = (gray < INK).astype(np.uint8) * 255
    lines = grid_lines(m)
    m[lines > 0] = 0
    m[lmask > 0] = 0
    # ต่อเส้นปากกาที่ถูกเส้นตารางตัดขาด (ปิดช่องว่างเฉพาะบริเวณเส้น)
    near = cv2.dilate(lines, np.ones((7, 7), np.uint8)) > 0
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m[near & (closed > 0)] = 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _merge(cl):
    """รวมก้อนหมึกที่ซ้อนกันแนวนอน (ตัวเลขเดียวที่เส้นขาด เช่น หัว 5 / ขีด 7)"""
    cl = sorted(cl, key=lambda c: c[0])
    out = []
    for c in cl:
        if out:
            p = out[-1]
            ov = min(p[0] + p[2], c[0] + c[2]) - max(p[0], c[0])
            if ov > 0.5 * min(p[2], c[2]):
                xa, ya = min(p[0], c[0]), min(p[1], c[1])
                xb, yb = max(p[0] + p[2], c[0] + c[2]), max(p[1] + p[3], c[1] + c[3])
                m = np.zeros((yb - ya, xb - xa), bool)
                for q in (p, c):
                    m[q[1] - ya:q[1] - ya + q[3], q[0] - xa:q[0] - xa + q[2]] |= q[4]
                out[-1] = [xa, ya, xb - xa, yb - ya, m]
                continue
        out.append(c)
    return out


def _split_wide(c, cw):
    """ก้อนกว้างเกิน (ตัวเลขเขียนติดกัน) -> แบ่งเท่า ๆ กันตามจำนวนช่องที่กิน"""
    n = max(2, int(round(c[2] / (1.2 * cw))))
    if c[2] < 1.8 * cw or n < 2:
        return [c]
    parts, step = [], c[2] / n
    for i in range(n):
        a, b = int(i * step), int((i + 1) * step)
        m = c[4][:, a:b]
        ys = np.nonzero(m.any(1))[0]
        if len(ys):
            parts.append([c[0] + a, c[1] + ys[0], b - a, ys[-1] - ys[0] + 1, m[ys[0]:ys[-1] + 1]])
    return parts


def _assign(rect, cl):
    """จัดตัวเลขเข้าช่องตามกติกาแบบฟอร์ม: จำนวนเต็มชิดขวาติดจุด, ทศนิยมชิดซ้ายต่อจากจุด
    -> ทนต่อการเขียนเยื้องทับเส้น (ค่าที่อ่านได้ไม่ขึ้นกับว่าเยื้องไปกี่ช่อง)"""
    x, y, w, h = rect
    cw = w / core.N_CELLS
    grid = [x + i * cw for i in range(core.N_CELLS + 1)]
    # เศษเส้นแนวตั้งที่เหลือ: บาง และอยู่ตรงตำแหน่งเส้นช่อง
    cl = [c for c in cl if not (c[2] <= 10 and c[3] >= 0.85 * h and min(abs(c[0] + c[2] / 2 - g) for g in grid) < 7)]
    # เศษกรอบรูปตัว L (หมึกบางมากเทียบกับกรอบ) ไม่ใช่ตัวเลข
    cl = [c for c in cl if c[4].sum() / (c[2] * c[3]) >= 0.1]
    # เศษเส้นแนวนอน: เตี้ยมาก
    cl = [c for c in cl if c[3] >= 0.3 * h]
    cl = [p for c in _merge(cl) for p in _split_wide(c, cw)]
    dot = x + (core.DECIMAL_IDX + 0.5) * cw
    left = sorted([c for c in cl if c[0] + c[2] / 2 < dot], key=lambda c: c[0])
    right = sorted([c for c in cl if c[0] + c[2] / 2 >= dot], key=lambda c: c[0])
    cells = [[] for _ in range(core.N_CELLS)]
    for j, c in enumerate(reversed(left[-core.DECIMAL_IDX:])):
        cells[core.DECIMAL_IDX - 1 - j].append(c)
    for j, c in enumerate(right[:core.N_CELLS - core.DECIMAL_IDX - 1]):
        cells[core.DECIMAL_IDX + 1 + j].append(c)
    return cells


def page_cells(gray, layout, lmask, up=0.9, down=0.5):
    """คืน {row_index: [crop หรือ None] x 10} สำหรับทุกแถวคำตอบในหน้า"""
    rows = [core.refine_rect(gray, r) for r in layout["rows"]]
    ink = _ink(gray, lmask)
    out = {i: [[] for _ in range(core.N_CELLS)] for i in range(len(rows))}
    comps = {i: [] for i in range(len(rows))}
    # ทำทีละคอลัมน์ (ซ้าย/ขวา)
    for col in (0, 1):
        idx = [i for i, r in enumerate(rows) if (r[0] > core.STD_W / 2) == bool(col)]
        if not idx:
            continue
        x0 = min(rows[i][0] for i in idx); x1 = max(rows[i][0] + rows[i][2] for i in idx)
        y0 = min(rows[i][1] for i in idx); y1 = max(rows[i][1] + rows[i][3] for i in idx)
        h = int(np.median([rows[i][3] for i in idx]))
        cw = np.median([rows[i][2] for i in idx]) / core.N_CELLS
        X0, X1 = int(x0 - 0.3 * cw), int(x1 + 0.6 * cw)
        Y0, Y1 = int(y0 - up * h), int(y1 + down * h)
        sub = ink[Y0:Y1, X0:X1]
        # จับกลุ่มด้วยภาพที่ขยายแล้ว (เส้นที่ขาดเล็กน้อยรวมเป็นตัวเดียว) แต่เก็บพิกเซลจริง
        grp = cv2.dilate(sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        n, lab, st, cen = cv2.connectedComponentsWithStats(grp)
        lab = np.where(sub > 0, lab, 0)
        for k in range(1, n):
            x, y, w, hh, _ = st[k]
            mk = lab[y:y + hh, x:x + w] == k
            a = int(mk.sum())                       # จำนวนพิกเซลหมึกจริง (ไม่นับส่วนที่ขยาย)
            if a < MIN_AREA:
                continue
            ys, xs = np.nonzero(mk)                 # กรอบจริงของหมึก
            y, x, hh, w = y + ys.min(), x + xs.min(), ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
            mk = mk[ys.min():ys.min() + hh, xs.min():xs.min() + w]
            if hh < 0.3 * h:
                continue
            cx, cy = x + w / 2 + X0, y + hh / 2 + Y0
            # แถว: จุดศูนย์กลางแนวตั้งใกล้ที่สุด (ตัวลอยขึ้นยังใกล้แถวตัวเองกว่าแถวบน)
            ri = min(idx, key=lambda i: abs(cy - (rows[i][1] + rows[i][3] / 2)))
            comps[ri].append([x + X0, y + Y0, w, hh, mk])
    for ri, cl in comps.items():
        out[ri] = _assign(rows[ri], cl)
    res = {}
    for ri, cells in out.items():
        row = []
        for comps in cells:
            if not comps:
                row.append(None); continue
            xa = min(c[0] for c in comps); ya = min(c[1] for c in comps)
            xb = max(c[0] + c[2] for c in comps); yb = max(c[1] + c[3] for c in comps)
            m = np.zeros((yb - ya, xb - xa), np.uint8)
            for x, y, w, hh, mk in comps:
                m[y - ya:y - ya + hh, x - xa:x - xa + w][mk] = 255
            row.append(m)
        res[ri] = row
    return res


def to32(mask, size=32):
    """มาสก์ตัวเลข -> 32x32 กลางภาพ (ตัวเลขขาวพื้นดำ แบบเดียวกับ prep_cell)"""
    s = max(mask.shape) + 8
    canvas = np.zeros((s, s), np.uint8)
    oy, ox = (s - mask.shape[0]) // 2, (s - mask.shape[1]) // 2
    canvas[oy:oy + mask.shape[0], ox:ox + mask.shape[1]] = mask
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


import re
_VALID = re.compile(r"^_*#*\.#*_*$")      # จำนวนเต็มติดกันชิดจุด + ทศนิยมติดกันต่อจากจุด


def page_cells_hybrid(gray, layout, lmask, ink_th=0.04):
    """ใช้วิธีเดิม (กรอบตายตัว) ถ้าผลดูเป็นคำตอบที่ถูกรูปแบบ ไม่งั้นสลับไปตัดตามหมึก
    วัดบนชุด hard case 5 แผ่น: เดิม 104/150 ข้อ -> hybrid 128/150 ข้อ (ไม่แย่ลงในเคสปกติ)
    คืน {row: (method, [cell: ภาพ gray ของช่องแบบเดิม / มาสก์หมึกแบบใหม่ / None])}"""
    ink_cells = None
    out = {}
    for ri, r in enumerate(layout["rows"]):
        cells = core.split_row(gray, r)
        has = [core.ink_ratio(c) >= ink_th for c in cells]
        pat = "".join("." if i == core.DECIMAL_IDX else ("#" if h else "_") for i, h in enumerate(has))
        if _VALID.match(pat):
            out[ri] = ("grid", [c if h and i != core.DECIMAL_IDX else None for i, (c, h) in enumerate(zip(cells, has))])
        else:
            if ink_cells is None:
                ink_cells = page_cells(gray, layout, lmask)
            out[ri] = ("ink", ink_cells[ri])
    return out
