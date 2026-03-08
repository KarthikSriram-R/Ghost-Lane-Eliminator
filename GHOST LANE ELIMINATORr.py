import cv2
import numpy as np
import time
import math

# ── YOLO (optional) ────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not found. Using contour fallback.")

# ── CONFIG ─────────────────────────────────────────────────────────────────
VEHICLE_CLASSES = {
    2: ("car",       2.0, (0,   220, 255)),
    3: ("motorbike", 1.0, (100, 255, 100)),
    5: ("bus",       5.0, (255, 160,  30)),
    7: ("truck",     5.0, ( 80, 100, 255)),
}
CONF_THRESHOLD = 0.20
MAX_GREEN      = 90
MIN_GREEN      = 12
PANEL_W        = 360
NUM_LANES      = 4
SPEED_ALPHA    = 0.3

# ── SCREEN CAPTURE CONFIG ──────────────────────────────────────────────────
USE_WEBCAM     = False           # Set to True to read from webcam pointed at screen
WEBCAM_ID      = 0

# !! IMPORTANT: If USE_WEBCAM=False, set SCREEN_REGION to your traffic video
# area ONLY (exclude the app window itself to avoid feedback loop).
# Format: (x, y, width, height)
# Example: (0, 0, 1280, 720) — top-left browser window
SCREEN_REGION  = (0, 0, 1280, 720)   # <-- adjust to your video area

# ── LANE ORIENTATION ───────────────────────────────────────────────────────
LANE_ORIENTATION = "horizontal"

# ── WINDOW NAME (single constant used everywhere) ─────────────────────────
WIN_NAME = "Ghost Lane Eliminator | AI4Dev'26 | PSG Tech"

# ── PALETTE ────────────────────────────────────────────────────────────────
BG        = (12,  16,  35)
ACCENT    = ( 0, 210, 255)
GREEN     = ( 0, 230,  80)
RED_C     = (50,  50, 220)
YELLOW_C  = ( 0, 190, 255)
WHITE     = (220, 230, 255)
DIM       = (70,  90, 130)
EMERGENCY = (40,  40, 230)
LANE_COLS = [
    (  0, 210, 255),
    (  0, 230,  80),
    (255, 160,  30),
    (200,  80, 255),
]
FONT = cv2.FONT_HERSHEY_DUPLEX


# ── HELPERS ────────────────────────────────────────────────────────────────
def pt(img, text, x, y, color=WHITE, scale=0.50, thick=1):
    cv2.putText(img, str(text), (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_bar(img, x, y, w, h, frac, color, bg=(18, 26, 52)):
    frac = min(max(frac, 0.0), 1.0)
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    fw = int(w * frac)
    if fw > 0:
        cv2.rectangle(img, (x, y), (x + fw, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), DIM, 1)


def hline(img, x, y, length):
    cv2.line(img, (x, y), (x + length, y), DIM, 1)


def traffic_light(img, x, y, state):
    cv2.rectangle(img, (x, y), (x + 28, y + 72), (18, 22, 48), -1)
    cv2.rectangle(img, (x, y), (x + 28, y + 72), DIM, 1)
    specs = [
        ((x + 14, y + 13), RED_C,    "red"),
        ((x + 14, y + 36), YELLOW_C, "yellow"),
        ((x + 14, y + 59), GREEN,    "green"),
    ]
    for center, on_color, s in specs:
        col = on_color if state == s else (18, 22, 48)
        cv2.circle(img, center, 9, col, -1)
        if state == s:
            ov = img.copy()
            cv2.circle(ov, center, 13, on_color, 1)
            cv2.addWeighted(ov, 0.3, img, 0.7, 0, img)


def calc_green(weighted):
    return min(int(weighted * 2 + MIN_GREEN), MAX_GREEN)


def iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / float(a1 + a2 - inter)


# ── SCREEN CAPTURE ─────────────────────────────────────────────────────────
def grab_screen(region=None):
    """Capture screen using MSS if available, else PIL."""
    try:
        import mss
        with mss.mss() as sct:
            mon = sct.monitors[1] if region is None else {
                "left": region[0], "top": region[1],
                "width": region[2], "height": region[3]
            }
            img = np.array(sct.grab(mon))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except ImportError:
        pass
    try:
        from PIL import ImageGrab
        bbox = None if region is None else tuple(region)
        img  = ImageGrab.grab(bbox=bbox)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except ImportError:
        print("[ERROR] Install mss or Pillow for screen capture: pip install mss")
        return None


# ── TRACKER ────────────────────────────────────────────────────────────────
class Tracker:
    def __init__(self, max_age=8):
        self.tracks  = {}
        self.next_id = 0
        self.max_age = max_age

    def update(self, dets):
        for tid in list(self.tracks):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        matched = set()
        out     = []

        for det in dets:
            box = det["box"]
            best_tid, best_score = None, 0.30
            for tid, tr in self.tracks.items():
                if tid in matched:
                    continue
                s = iou(box, tr["box"])
                if s > best_score:
                    best_score, best_tid = s, tid

            if best_tid is not None:
                pb  = self.tracks[best_tid]["box"]
                dx  = ((box[0] + box[2]) - (pb[0] + pb[2])) / 2.0
                dy  = ((box[1] + box[3]) - (pb[1] + pb[3])) / 2.0
                raw = math.hypot(dx, dy)
                prv = self.tracks[best_tid].get("speed", 0.0)
                spd = SPEED_ALPHA * raw + (1.0 - SPEED_ALPHA) * prv
                self.tracks[best_tid].update(box=box, age=0, speed=spd)
                matched.add(best_tid)
                out.append(dict(det, tid=best_tid, speed=spd))
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": box, "age": 0, "speed": 0.0}
                out.append(dict(det, tid=tid, speed=0.0))

        return out


# ── LANE MANAGER ───────────────────────────────────────────────────────────
class LaneManager:
    def __init__(self, n=4):
        self.n           = n
        self.weighted    = [0.0]   * n
        self.emergency   = [False] * n
        self.green_times = [MIN_GREEN] * n
        self.active      = 0
        self.timer       = time.time()

    def assign(self, dets, frame_h, frame_w):
        self.weighted  = [0.0]   * self.n
        self.emergency = [False] * self.n

        if LANE_ORIENTATION == "horizontal":
            strip = frame_h / float(self.n)
            for d in dets:
                cy   = (d["box"][1] + d["box"][3]) / 2.0
                lane = min(int(cy / strip), self.n - 1)
                d["lane"] = lane
                self.weighted[lane] += d["weight"]
                if d["emergency"]:
                    self.emergency[lane] = True
        else:
            strip = frame_w / float(self.n)
            for d in dets:
                cx   = (d["box"][0] + d["box"][2]) / 2.0
                lane = min(int(cx / strip), self.n - 1)
                d["lane"] = lane
                self.weighted[lane] += d["weight"]
                if d["emergency"]:
                    self.emergency[lane] = True

        self.green_times = [calc_green(w) for w in self.weighted]

    def tick(self):
        for i, e in enumerate(self.emergency):
            if e:
                self.active = i
                self.timer  = time.time()
                return
        if time.time() - self.timer >= self.green_times[self.active]:
            self.active = (self.active + 1) % self.n
            self.timer  = time.time()

    def remaining(self):
        return max(0.0, self.green_times[self.active] - (time.time() - self.timer))


# ── HEATMAP ────────────────────────────────────────────────────────────────
class HeatMap:
    def __init__(self, h, w):
        self.hmap = np.zeros((h, w), dtype=np.float32)

    def update(self, dets):
        self.hmap *= 0.96
        for d in dets:
            cx = (d["box"][0] + d["box"][2]) // 2
            cy = (d["box"][1] + d["box"][3]) // 2
            cv2.circle(self.hmap, (cx, cy), 40, 1.0, -1)

    def overlay(self, frame):
        norm = cv2.normalize(self.hmap, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.65, heat, 0.35, 0)


# ── SESSION STATS ──────────────────────────────────────────────────────────
class Stats:
    def __init__(self):
        self.t0        = time.time()
        self.peak      = 0.0
        self.emg_count = 0

    def update(self, total_w, dets):
        if total_w > self.peak:
            self.peak = total_w
        for d in dets:
            if d["emergency"]:
                self.emg_count += 1

    def uptime(self):
        s = int(time.time() - self.t0)
        return "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


# ── CONTOUR FALLBACK ───────────────────────────────────────────────────────
def detect_contour(frame):
    out = []
    lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a_norm = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    color_mask = cv2.bitwise_or(
        cv2.inRange(a_norm, 140, 255),
        cv2.inRange(b_norm, 140, 255),
    )
    color_mask = cv2.bitwise_or(
        color_mask,
        cv2.inRange(a_norm, 0, 110),
    )
    bright_mask = cv2.inRange(l, 200, 255)
    combined    = cv2.bitwise_or(color_mask, bright_mask)
    kernel   = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if 200 < area < 8000:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(h, 1)
            if 0.3 < aspect < 4.5:
                out.append({
                    "box":       (x, y, x + w, y + h),
                    "label_raw": "car",
                    "label":     "car",
                    "weight":    2.0,
                    "color":     ACCENT,
                    "emergency": False,
                })
    return out


# ── DRAW LANE DIVIDERS ─────────────────────────────────────────────────────
def draw_lanes(display, h, w, active, lane_cols):
    strip_h = h // NUM_LANES
    strip_w = w // NUM_LANES

    if LANE_ORIENTATION == "horizontal":
        for i in range(1, NUM_LANES):
            yd = i * strip_h
            lc = lane_cols[i - 1] if (i - 1) == active else DIM
            cv2.line(display, (0, yd), (w, yd), lc, 1)
            pt(display, "L%d" % i, 4, yd - 4, lc, 0.38)
        yt = active * strip_h
        ov = display.copy()
        cv2.rectangle(ov, (0, yt), (w, yt + strip_h), lane_cols[active], 1)
        cv2.addWeighted(ov, 0.15, display, 0.85, 0, display)
        pt(display, "ACTIVE", 4, yt + 14, lane_cols[active], 0.42)
    else:
        for i in range(1, NUM_LANES):
            xd = i * strip_w
            lc = lane_cols[i - 1] if (i - 1) == active else DIM
            cv2.line(display, (xd, 0), (xd, h), lc, 1)
            pt(display, "L%d" % i, xd + 4, 14, lc, 0.38)
        xl = active * strip_w
        ov = display.copy()
        cv2.rectangle(ov, (xl, 0), (xl + strip_w, h), lane_cols[active], 1)
        cv2.addWeighted(ov, 0.15, display, 0.85, 0, display)
        pt(display, "ACTIVE", xl + 4, 28, lane_cols[active], 0.42)


# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    global LANE_ORIENTATION

    model = None
    if YOLO_AVAILABLE:
        print("[INFO] Loading YOLOv8n ...")
        try:
            model = YOLO("yolov8n.pt")
            print("[INFO] YOLOv8n ready.")
        except Exception as e:
            print("[WARNING] YOLO load failed:", e)

    # ── Source setup
    cap = None
    if USE_WEBCAM:
        cap = cv2.VideoCapture(WEBCAM_ID)
        if not cap.isOpened():
            print("[ERROR] Webcam %d not found. Switching to screen capture." % WEBCAM_ID)
            cap = None
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print("[INFO] Webcam ready.")
    else:
        print("[INFO] Screen capture mode — region:", SCREEN_REGION)

    # ── Get frame size
    if cap is not None:
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
    else:
        test = grab_screen(SCREEN_REGION)
        if test is None:
            return
        cam_h, cam_w = test.shape[:2]

    tracker = Tracker()
    lanes   = LaneManager(NUM_LANES)
    heatmap = HeatMap(cam_h, cam_w)
    stats   = Stats()

    fps        = 0.0
    fps_timer  = time.time()
    frame_n    = 0
    show_heat  = False
    paused     = False
    shot_n     = 0
    last_frame = None

    # ── FIX: Create ONE named window before the loop ───────────────────────
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, cam_w + PANEL_W, cam_h)

    # In screen-capture mode, keep preview window outside the capture region
    # to prevent recursive capture ("screen inside screen" effect).
    if cap is None and SCREEN_REGION is not None and len(SCREEN_REGION) == 4:
        rx, ry, rw, rh = SCREEN_REGION
        target_x = max(0, rx + rw + 20)
        target_y = max(0, ry)
        try:
            cv2.moveWindow(WIN_NAME, target_x, target_y)
        except Exception:
            pass
    # ───────────────────────────────────────────────────────────────────────

    print("[INFO] Controls: Q=quit  R=reset  S=screenshot  H=heatmap  SPACE=pause  1-4=force lane  O=orientation")
    print("[INFO] Source:   Webcam=%s  YOLO=%s  Orientation=%s" % (USE_WEBCAM, YOLO_AVAILABLE, LANE_ORIENTATION))

    while True:
        # ── Grab frame
        if not paused:
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Lost camera.")
                    break
            else:
                frame = grab_screen(SCREEN_REGION)
                if frame is None:
                    break

                # ── FIX: Skip frames that contain our own window
                # (prevents screen-capture feedback loop)
                if frame.shape[0] < 10 or frame.shape[1] < 10:
                    continue

            last_frame = frame.copy()
        else:
            if last_frame is None:
                continue
            frame = last_frame.copy()

        frame_n += 1
        h, w = frame.shape[:2]

        # ── Resize heatmap if frame size changed
        if heatmap.hmap.shape != (h, w):
            heatmap = HeatMap(h, w)

        # ── Detect
        raw = []
        if model is not None:
            res = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            for box in res.boxes:
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                name = model.names[cid].lower()
                is_v = cid in VEHICLE_CLASSES
                is_e = "ambulance" in name
                if is_v or is_e:
                    if is_e:
                        lraw, wt, col = "ambulance", 0.0, EMERGENCY
                    else:
                        lraw, wt, col = VEHICLE_CLASSES[cid]
                    raw.append({
                        "box":       (x1, y1, x2, y2),
                        "label_raw": lraw,
                        "label":     "%s %.0f%%" % (lraw, conf * 100),
                        "weight":    wt,
                        "color":     col,
                        "emergency": is_e,
                    })
        else:
            raw = detect_contour(frame)

        # ── Track + assign
        tracked = tracker.update(raw)
        lanes.assign(tracked, h, w)
        lanes.tick()
        heatmap.update(tracked)

        total_w = sum(lanes.weighted)
        stats.update(total_w, tracked)

        # ── FPS
        if frame_n % 15 == 0:
            now       = time.time()
            fps       = 15.0 / max(now - fps_timer, 0.001)
            fps_timer = now

        # ── Draw
        display = heatmap.overlay(frame) if show_heat else frame.copy()
        draw_lanes(display, h, w, lanes.active, LANE_COLS)

        for d in tracked:
            x1, y1, x2, y2 = d["box"]
            col   = d["color"]
            thick = 3 if d["emergency"] else 2
            cv2.rectangle(display, (x1, y1), (x2, y2), col, thick)
            spd = d.get("speed", 0.0)
            lbl = "%s %.0fpx" % (d["label"], spd)
            lw  = len(lbl) * 8 + 6
            ly  = max(y1 - 22, 0)
            cv2.rectangle(display, (x1, ly), (x1 + lw, y1), col, -1)
            pt(display, lbl, x1 + 3, y1 - 6, BG, 0.44)
            pt(display, "#%d" % d["tid"], x1 + 2, y2 - 4, col, 0.38)

        if paused:
            cv2.rectangle(display, (w//2-60, h//2-18), (w//2+60, h//2+18), BG, -1)
            pt(display, "PAUSED", w//2 - 42, h//2 + 8, YELLOW_C, 0.70, 2)

        # ── Panel
        canvas = np.full((h, w + PANEL_W, 3), BG, dtype=np.uint8)
        canvas[:h, :w] = display
        cv2.line(canvas, (w, 0), (w, h), DIM, 1)

        px = w + 12
        pw = PANEL_W - 20
        py = 12

        pt(canvas, "GHOST LANE",           px, py + 16, ACCENT, 0.72, 2)
        pt(canvas, "ELIMINATOR  v2.0",     px, py + 33, ACCENT, 0.50)
        pt(canvas, "AI4Dev'26 | PSG Tech", px, py + 47, DIM,    0.38)
        hline(canvas, px, py + 54, pw)
        py += 62

        emg_any   = any(lanes.emergency)
        has_veh   = total_w > 0
        sig_state = "green" if (has_veh and not emg_any) else "red"
        traffic_light(canvas, px, py, sig_state)

        lx = px + 38
        if emg_any:
            sl, sc = "EMERGENCY!", EMERGENCY
        elif has_veh:
            sl, sc = "SIGNAL ON",  GREEN
        else:
            sl, sc = "NO TRAFFIC", DIM

        pt(canvas, sl, lx, py + 14, sc, 0.60, 2)
        pt(canvas, "Lane %d  |  %.0fs left" % (lanes.active + 1, lanes.remaining()),
           lx, py + 30, WHITE, 0.44)
        pt(canvas, "Green: %ds   Load: %.1f" % (lanes.green_times[lanes.active], total_w),
           lx, py + 45, WHITE, 0.44)
        py += 84

        hline(canvas, px, py, pw)
        py += 8

        pt(canvas, "LANE OVERVIEW", px, py + 12, DIM, 0.40)
        py += 18

        for i in range(NUM_LANES):
            lc   = LANE_COLS[i]
            wt_i = lanes.weighted[i]
            gt_i = lanes.green_times[i]
            mark = "> " if i == lanes.active else "  "
            emgm = " !" if lanes.emergency[i] else ""
            txt  = "%sL%d  %.1fu  %ds%s" % (mark, i + 1, wt_i, gt_i, emgm)
            tc   = lc if i == lanes.active else WHITE
            pt(canvas, txt, px, py + 11, tc, 0.47)
            draw_bar(canvas, px, py + 14, pw, 7, gt_i / float(MAX_GREEN), lc)
            py += 26

        hline(canvas, px, py, pw)
        py += 10

        pt(canvas, "CONGESTION INDEX", px, py + 12, DIM, 0.40)
        py += 18
        cong     = min(total_w / 20.0, 1.0)
        cong_col = GREEN if cong < 0.4 else (YELLOW_C if cong < 0.75 else RED_C)
        cong_lbl = "LOW" if cong < 0.4 else ("MEDIUM" if cong < 0.75 else "HIGH")
        draw_bar(canvas, px, py, pw - 70, 14, cong, cong_col)
        pt(canvas, "%s  %.0f%%" % (cong_lbl, cong * 100),
           px + pw - 66, py + 11, cong_col, 0.44)
        py += 26

        hline(canvas, px, py, pw)
        py += 10

        pt(canvas, "SESSION STATS", px, py + 12, DIM, 0.40)
        py += 20

        half = pw // 2
        stat_pairs = [
            ("Uptime",    stats.uptime(),        WHITE,
             "Peak load", "%.1f u" % stats.peak, YELLOW_C),
            ("FPS",       "%.1f" % fps,           ACCENT,
             "Emergency", str(stats.emg_count),   EMERGENCY if stats.emg_count else DIM),
            ("Vehicles",  str(len(tracked)),       GREEN,
             "Orient",    LANE_ORIENTATION[:4].upper(), DIM),
        ]
        for lk, lv, lc2, rk, rv, rc in stat_pairs:
            pt(canvas, lk, px,        py + 11, DIM, 0.38)
            pt(canvas, lv, px,        py + 24, lc2, 0.48)
            pt(canvas, rk, px + half, py + 11, DIM, 0.38)
            pt(canvas, rv, px + half, py + 24, rc,  0.48)
            py += 36

        mode = "YOLOv8n" if model else "Contour"
        src  = "Webcam" if cap else "Screen"
        heat = "ON" if show_heat else "OFF"
        hline(canvas, px, h - 22, pw)
        pt(canvas, "%s|%s  Heat:%s  F:%d" % (mode, src, heat, frame_n),
           px, h - 8, DIM, 0.37)

        # ── FIX: Always draw to the SAME named window ──────────────────────
        cv2.imshow(WIN_NAME, canvas)
        # ───────────────────────────────────────────────────────────────────

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quit.")
            break
        elif key == ord('r'):
            stats = Stats()
            print("[INFO] Stats reset.")
        elif key == ord('s'):
            fn = "gle_shot_%03d.png" % shot_n
            cv2.imwrite(fn, canvas)
            shot_n += 1
            print("[INFO] Saved:", fn)
        elif key == ord('h'):
            show_heat = not show_heat
            print("[INFO] Heatmap:", "ON" if show_heat else "OFF")
        elif key == ord(' '):
            paused = not paused
            print("[INFO]", "Paused" if paused else "Resumed")
        elif key == ord('o'):
            LANE_ORIENTATION = "vertical" if LANE_ORIENTATION == "horizontal" else "horizontal"
            print("[INFO] Orientation:", LANE_ORIENTATION)
        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            lanes.active = key - ord('1')
            lanes.timer  = time.time()
            print("[INFO] Manual lane:", lanes.active + 1)

        # ── FIX: Exit cleanly if window is closed by user ──────────────────
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Window closed by user.")
            break
        # ───────────────────────────────────────────────────────────────────

    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()