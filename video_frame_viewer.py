# -*- coding: utf-8 -*-
"""
video_frame_viewer.py

功能：
1. 讀取一般影片檔，例如 mp4 / avi / mov
2. 使用滑桿逐幀查看影片
3. 使用鍵盤左右鍵或 A/D 切換上一幀、下一幀
4. 按按鈕或快捷鍵儲存目前畫面
5. 可選擇儲存「原始影格」或「含資訊列的顯示畫面」

執行：
python video_frame_viewer.py

需要安裝：
pip install opencv-python numpy
"""

import time
from pathlib import Path

import cv2
import numpy as np


# =========================
# 基本設定
# =========================

VIDEO_PATH = r"C:\Users\mc2\Desktop\ti_tracking\video\TI_貼桌online.mkv"

SAVE_DIR = Path("video_snapshots")

WINDOW_NAME = "Video Frame Viewer"
TRACKBAR_NAME = "Frame"

# 顯示尺寸設定
MAX_DISPLAY_WIDTH = 960
MAX_DISPLAY_HEIGHT = 540

TITLE_BAR_HEIGHT = 42
CONTROL_HEIGHT = 68
BORDER = 12

BUTTON_HEIGHT = 34
BUTTON_WIDTH = 150
BUTTON_GAP = 14


# =========================
# 狀態類別
# =========================

class ViewerState:
    def __init__(self):
        self.button_rects = {}
        self.last_canvas = None
        self.last_raw_frame = None
        self.last_frame_idx = 0
        self.is_playing = False


# =========================
# 工具函式
# =========================

def resize_keep_ratio(frame, max_w=MAX_DISPLAY_WIDTH, max_h=MAX_DISPLAY_HEIGHT):
    """等比例縮放影片畫面，避免太大超出螢幕。"""
    h, w = frame.shape[:2]

    scale = min(max_w / w, max_h / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)

    if scale == 1.0:
        return frame.copy(), scale

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def draw_button(canvas, rect, label, active=True):
    x, y, w, h = rect

    fill = (64, 64, 64) if active else (38, 38, 38)
    border = (150, 150, 150) if active else (80, 80, 80)
    text_color = (255, 255, 255) if active else (140, 140, 140)

    cv2.rectangle(canvas, (x, y), (x + w, y + h), fill, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2 - 2

    cv2.putText(
        canvas,
        label,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        text_color,
        1,
        cv2.LINE_AA,
    )


def point_in_rect(x, y, rect):
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def format_time_by_frame(frame_idx, fps):
    if fps <= 0:
        return "00:00.000"

    total_sec = frame_idx / fps
    minutes = int(total_sec // 60)
    seconds = int(total_sec % 60)
    millis = int((total_sec - int(total_sec)) * 1000)

    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def read_frame(cap, frame_idx):
    """
    讀取指定 frame。
    注意：對長影片頻繁跳轉可能會受影片編碼影響而稍慢，這是正常現象。
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()

    if not ok or frame is None:
        return None

    return frame


def compose_view(frame, frame_idx, num_frames, fps, state=None):
    """把影片畫面、標題列、控制按鈕合成成一張 canvas。"""
    display_frame, scale = resize_keep_ratio(frame)

    frame_h, frame_w = display_frame.shape[:2]

    canvas_w = max(frame_w + BORDER * 2, 720)
    canvas_h = TITLE_BAR_HEIGHT + BORDER + frame_h + BORDER + CONTROL_HEIGHT

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 標題資訊
    time_text = format_time_by_frame(frame_idx, fps)
    title = (
        f"Frame {frame_idx + 1}/{num_frames}  |  "
        f"Time {time_text}  |  "
        f"FPS {fps:.2f}"
    )

    cv2.putText(
        canvas,
        title,
        (BORDER, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    # 影片畫面置中
    frame_x = (canvas_w - frame_w) // 2
    frame_y = TITLE_BAR_HEIGHT + BORDER

    canvas[frame_y:frame_y + frame_h, frame_x:frame_x + frame_w] = display_frame
    cv2.rectangle(
        canvas,
        (frame_x - 1, frame_y - 1),
        (frame_x + frame_w, frame_y + frame_h),
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )

    # 按鈕
    button_y = TITLE_BAR_HEIGHT + BORDER + frame_h + BORDER + 14

    total_button_w = BUTTON_WIDTH * 3 + BUTTON_GAP * 2
    button_x0 = (canvas_w - total_button_w) // 2

    button_rects = {
        "save_raw": (button_x0, button_y, BUTTON_WIDTH, BUTTON_HEIGHT),
        "save_view": (
            button_x0 + BUTTON_WIDTH + BUTTON_GAP,
            button_y,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
        ),
        "save_both": (
            button_x0 + (BUTTON_WIDTH + BUTTON_GAP) * 2,
            button_y,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
        ),
    }

    draw_button(canvas, button_rects["save_raw"], "Save Raw")
    draw_button(canvas, button_rects["save_view"], "Save View")
    draw_button(canvas, button_rects["save_both"], "Save Both")

    # 底部提示
    hint = "Keys: A/Left previous | D/Right next | Space play/pause | S save raw | V save view | B save both | Q/ESC quit"
    cv2.putText(
        canvas,
        hint,
        (BORDER, canvas_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    if state is not None:
        state.button_rects = button_rects
        state.last_canvas = canvas.copy()
        state.last_raw_frame = frame.copy()
        state.last_frame_idx = frame_idx

    return canvas


def save_current_frame(state, save_raw=True, save_view=False):
    """儲存目前幀。raw 是影片原始畫面，view 是包含資訊列與按鈕的畫面。"""
    if state.last_raw_frame is None:
        print("[WARN] No frame to save.")
        return []

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp = f"{timestamp}_{int(time.time() * 1000) % 1000:03d}"

    saved_paths = []
    frame_idx = state.last_frame_idx

    if save_raw:
        path = SAVE_DIR / f"frame_{frame_idx:06d}_raw_{timestamp}.png"
        cv2.imwrite(str(path), state.last_raw_frame)
        saved_paths.append(path)

    if save_view:
        path = SAVE_DIR / f"frame_{frame_idx:06d}_view_{timestamp}.png"
        cv2.imwrite(str(path), state.last_canvas)
        saved_paths.append(path)

    if saved_paths:
        print("[SAVE] " + ", ".join(str(p) for p in saved_paths))

    return saved_paths


# =========================
# 主程式
# =========================

def main():
    video_path = Path(VIDEO_PATH)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Cannot find video file:\n{video_path}\n\n"
            "請先修改程式最上方的 VIDEO_PATH。"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if num_frames <= 0:
        raise RuntimeError("Cannot read video frame count.")

    if fps <= 0:
        fps = 30.0

    state = ViewerState()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def update(_=None):
        idx = cv2.getTrackbarPos(TRACKBAR_NAME, WINDOW_NAME)
        idx = min(max(idx, 0), num_frames - 1)

        frame = read_frame(cap, idx)

        if frame is None:
            print(f"[WARN] Cannot read frame {idx}")
            return

        canvas = compose_view(frame, idx, num_frames, fps, state=state)
        cv2.imshow(WINDOW_NAME, canvas)

    def set_frame(idx):
        idx = min(max(idx, 0), num_frames - 1)
        cv2.setTrackbarPos(TRACKBAR_NAME, WINDOW_NAME, idx)

    def next_frame(step=1):
        idx = cv2.getTrackbarPos(TRACKBAR_NAME, WINDOW_NAME)
        set_frame(idx + step)

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if point_in_rect(x, y, state.button_rects.get("save_raw", (0, 0, 0, 0))):
            save_current_frame(state, save_raw=True, save_view=False)

        elif point_in_rect(x, y, state.button_rects.get("save_view", (0, 0, 0, 0))):
            save_current_frame(state, save_raw=False, save_view=True)

        elif point_in_rect(x, y, state.button_rects.get("save_both", (0, 0, 0, 0))):
            save_current_frame(state, save_raw=True, save_view=True)

    cv2.createTrackbar(TRACKBAR_NAME, WINDOW_NAME, 0, max(num_frames - 1, 1), update)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    update()

    print("====================================")
    print("Video Frame Viewer")
    print("====================================")
    print(f"Video path : {video_path}")
    print(f"Frames     : {num_frames}")
    print(f"FPS        : {fps:.2f}")
    print(f"Save dir   : {SAVE_DIR.resolve()}")
    print("Keys:")
    print("  q / ESC      : quit")
    print("  left / a     : previous frame")
    print("  right / d    : next frame")
    print("  space        : play / pause")
    print("  s            : save raw frame")
    print("  v            : save view image")
    print("  b            : save both")
    print("====================================")

    while True:
        # 播放中用較短 wait，暫停時也讓 GUI 持續刷新
        delay = max(1, int(1000 / fps)) if state.is_playing else 30
        key = cv2.waitKeyEx(delay)

        if key in (27, ord("q")):
            break

        elif key in (ord("a"), 2424832):
            state.is_playing = False
            next_frame(-1)

        elif key in (ord("d"), 2555904):
            state.is_playing = False
            next_frame(1)

        elif key == ord(" "):
            state.is_playing = not state.is_playing
            print("[PLAY]" if state.is_playing else "[PAUSE]")

        elif key == ord("s"):
            save_current_frame(state, save_raw=True, save_view=False)

        elif key == ord("v"):
            save_current_frame(state, save_raw=False, save_view=True)

        elif key == ord("b"):
            save_current_frame(state, save_raw=True, save_view=True)

        if state.is_playing:
            idx = cv2.getTrackbarPos(TRACKBAR_NAME, WINDOW_NAME)
            if idx >= num_frames - 1:
                state.is_playing = False
                set_frame(num_frames - 1)
            else:
                next_frame(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
