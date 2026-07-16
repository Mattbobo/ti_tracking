# -*- coding: utf-8 -*-
import time
from pathlib import Path

import cv2
import h5py
import numpy as np


H5_PATH = r"C:\Users\mc2\Desktop\ti_tracking\Record\straight\data_20260511_170754.h5"
DATASET_NAME = "DS1"

# DS1.shape = (frames, channels, H, W)
RDI_CHANNEL = 0
AZIMUTH_CHANNEL = 1

TILE_WIDTH = 240
TILE_HEIGHT = 240
TILE_GAP = 18
TILE_BORDER = 12
TITLE_BAR_HEIGHT = 38
CONTROL_HEIGHT = 64
COLORMAP = cv2.COLORMAP_JET
SAVE_DIR = Path("h5_snapshots")

# The previous display was vertically inverted. Keep row 0 at the bottom.
FLIP_VERTICAL = True

# Set these to fixed values if you want a stable color scale.
RDI_VMIN = None
RDI_VMAX = None
AZIMUTH_VMIN = None
AZIMUTH_VMAX = None

BUTTON_HEIGHT = 34
BUTTON_WIDTH = 140
BUTTON_GAP = 12


class ViewerState:
    def __init__(self):
        self.button_rects = {}
        self.last_canvas = None


def normalize_to_uint8(img, vmin=None, vmax=None):
    img = np.asarray(img, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    if vmin is None or vmax is None:
        vmin = float(np.percentile(img, 2))
        vmax = float(np.percentile(img, 98))

    if vmax <= vmin:
        vmax = vmin + 1.0

    img = np.clip(img, vmin, vmax)
    return ((img - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)


def make_heatmap(img, vmin=None, vmax=None, size=(TILE_WIDTH, TILE_HEIGHT)):
    if FLIP_VERTICAL:
        img = np.flipud(img)

    img_u8 = normalize_to_uint8(img, vmin=vmin, vmax=vmax)
    img_u8 = cv2.resize(img_u8, size, interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(img_u8, COLORMAP)
    return heatmap


def unshift_azimuth(azimuth):
    return np.fft.ifftshift(azimuth, axes=1)


def get_frame_tiles(ds, frame_idx):
    rdi = ds[frame_idx, RDI_CHANNEL, :, :].astype(np.float32)
    azimuth = ds[frame_idx, AZIMUTH_CHANNEL, :, :].astype(np.float32)
    azimuth_unshifted = unshift_azimuth(azimuth)

    rdi_view = make_heatmap(rdi, vmin=RDI_VMIN, vmax=RDI_VMAX)
    azimuth_view = make_heatmap(azimuth, vmin=AZIMUTH_VMIN, vmax=AZIMUTH_VMAX)
    azimuth_unshifted_view = make_heatmap(
        azimuth_unshifted,
        vmin=AZIMUTH_VMIN,
        vmax=AZIMUTH_VMAX,
    )
    return rdi_view, azimuth_view, azimuth_unshifted_view


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


def compose_view(ds, frame_idx, state=None):
    rdi_view, azimuth_view, azimuth_unshifted_view = get_frame_tiles(ds, frame_idx)

    grid_w = TILE_WIDTH * 3 + TILE_GAP * 2
    canvas_w = max(TILE_BORDER * 2 + grid_w, 760)
    canvas_h = TITLE_BAR_HEIGHT + TILE_BORDER + TILE_HEIGHT + TILE_BORDER + CONTROL_HEIGHT
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    grid_x0 = (canvas_w - grid_w) // 2
    grid_y0 = TITLE_BAR_HEIGHT + TILE_BORDER
    positions = [
        (grid_x0, grid_y0),
        (grid_x0 + TILE_WIDTH + TILE_GAP, grid_y0),
        (grid_x0 + (TILE_WIDTH + TILE_GAP) * 2, grid_y0),
    ]

    for tile, (x, y) in zip((rdi_view, azimuth_view, azimuth_unshifted_view), positions):
        canvas[y:y + TILE_HEIGHT, x:x + TILE_WIDTH] = tile
        cv2.rectangle(
            canvas,
            (x - 1, y - 1),
            (x + TILE_WIDTH, y + TILE_HEIGHT),
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

    button_y = TITLE_BAR_HEIGHT + TILE_BORDER + TILE_HEIGHT + TILE_BORDER + 12
    total_button_w = BUTTON_WIDTH * 4 + BUTTON_GAP * 3
    button_x0 = (canvas_w - total_button_w) // 2
    button_rects = {
        "save_rdi": (button_x0, button_y, BUTTON_WIDTH, BUTTON_HEIGHT),
        "save_azimuth": (
            button_x0 + BUTTON_WIDTH + BUTTON_GAP,
            button_y,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
        ),
        "save_azimuth_unshifted": (
            button_x0 + (BUTTON_WIDTH + BUTTON_GAP) * 2,
            button_y,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
        ),
        "save_all": (
            button_x0 + (BUTTON_WIDTH + BUTTON_GAP) * 3,
            button_y,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
        ),
    }

    draw_button(canvas, button_rects["save_rdi"], "Save RDI")
    draw_button(canvas, button_rects["save_azimuth"], "Save Azimuth")
    draw_button(canvas, button_rects["save_azimuth_unshifted"], "Save Az Unshift")
    draw_button(canvas, button_rects["save_all"], "Save All")

    if state is not None:
        state.button_rects = button_rects
        state.last_canvas = canvas.copy()

    return canvas


def save_current_images(
    ds,
    frame_idx,
    save_rdi=True,
    save_azimuth=True,
    save_azimuth_unshifted=False,
):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    rdi_view, azimuth_view, azimuth_unshifted_view = get_frame_tiles(ds, frame_idx)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp = f"{timestamp}_{int(time.time() * 1000) % 1000:03d}"
    saved_paths = []

    if save_rdi:
        path = SAVE_DIR / f"frame_{frame_idx:05d}_rdi_{timestamp}.png"
        cv2.imwrite(str(path), rdi_view)
        saved_paths.append(path)

    if save_azimuth:
        path = SAVE_DIR / f"frame_{frame_idx:05d}_azimuth_{timestamp}.png"
        cv2.imwrite(str(path), azimuth_view)
        saved_paths.append(path)

    if save_azimuth_unshifted:
        path = SAVE_DIR / f"frame_{frame_idx:05d}_azimuth_unshifted_{timestamp}.png"
        cv2.imwrite(str(path), azimuth_unshifted_view)
        saved_paths.append(path)

    return saved_paths


def point_in_rect(x, y, rect):
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def main():
    with h5py.File(H5_PATH, "r") as f:
        if DATASET_NAME not in f:
            raise KeyError(f"Cannot find dataset '{DATASET_NAME}' in {H5_PATH}")

        ds = f[DATASET_NAME]

        if len(ds.shape) != 4:
            raise ValueError(f"Expected DS shape = (N, C, H, W), got {ds.shape}")

        num_frames, num_channels, _, _ = ds.shape

        for channel_name, channel_idx in (
            ("RDI_CHANNEL", RDI_CHANNEL),
            ("AZIMUTH_CHANNEL", AZIMUTH_CHANNEL),
        ):
            if channel_idx < 0 or channel_idx >= num_channels:
                raise ValueError(
                    f"{channel_name}={channel_idx} out of range. DS shape={ds.shape}"
                )

        window_name = "H5 RDI + Azimuth"
        trackbar_name = "Frame"
        state = ViewerState()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            window_name,
            max(TILE_BORDER * 2 + TILE_WIDTH * 3 + TILE_GAP * 2, 760),
            TITLE_BAR_HEIGHT + TILE_BORDER + TILE_HEIGHT + TILE_BORDER + CONTROL_HEIGHT,
        )

        def update(_=None):
            idx = cv2.getTrackbarPos(trackbar_name, window_name)
            idx = min(idx, num_frames - 1)
            cv2.imshow(window_name, compose_view(ds, idx, state=state))

        def save_and_update(
            save_rdi=True,
            save_azimuth=True,
            save_azimuth_unshifted=False,
        ):
            idx = cv2.getTrackbarPos(trackbar_name, window_name)
            idx = min(idx, num_frames - 1)
            saved_paths = save_current_images(
                ds,
                idx,
                save_rdi=save_rdi,
                save_azimuth=save_azimuth,
                save_azimuth_unshifted=save_azimuth_unshifted,
            )
            print(f"[SAVE] frame={idx}: " + ", ".join(str(path) for path in saved_paths))
            update()

        def on_mouse(event, x, y, _flags, _param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return

            if point_in_rect(x, y, state.button_rects.get("save_rdi", (0, 0, 0, 0))):
                save_and_update(save_rdi=True, save_azimuth=False)
            elif point_in_rect(x, y, state.button_rects.get("save_azimuth", (0, 0, 0, 0))):
                save_and_update(save_rdi=False, save_azimuth=True)
            elif point_in_rect(
                x,
                y,
                state.button_rects.get("save_azimuth_unshifted", (0, 0, 0, 0)),
            ):
                save_and_update(
                    save_rdi=False,
                    save_azimuth=False,
                    save_azimuth_unshifted=True,
                )
            elif point_in_rect(x, y, state.button_rects.get("save_all", (0, 0, 0, 0))):
                save_and_update(
                    save_rdi=True,
                    save_azimuth=True,
                    save_azimuth_unshifted=True,
                )

        cv2.createTrackbar(trackbar_name, window_name, 0, max(num_frames - 1, 1), update)
        cv2.setMouseCallback(window_name, on_mouse)
        update()

        print("====================================")
        print("H5 RDI + Azimuth slider viewer")
        print("====================================")
        print(f"H5 path : {H5_PATH}")
        print(f"Dataset : {DATASET_NAME}")
        print(f"Shape   : {ds.shape}")
        print(f"RDI ch  : {RDI_CHANNEL}")
        print(f"Az ch   : {AZIMUTH_CHANNEL}")
        print("Keys:")
        print("  q / ESC      : quit")
        print("  left / a     : previous frame")
        print("  right / d    : next frame")
        print("  1            : save RDI image")
        print("  2            : save Azimuth image")
        print("  3            : save unshifted Azimuth image")
        print("  s            : save all images")
        print("====================================")

        while True:
            key = cv2.waitKeyEx(30)

            if key in (27, ord("q")):
                break

            if key in (ord("a"), 2424832):
                idx = cv2.getTrackbarPos(trackbar_name, window_name)
                cv2.setTrackbarPos(trackbar_name, window_name, max(idx - 1, 0))

            elif key in (ord("d"), 2555904):
                idx = cv2.getTrackbarPos(trackbar_name, window_name)
                cv2.setTrackbarPos(trackbar_name, window_name, min(idx + 1, num_frames - 1))

            elif key == ord("1"):
                save_and_update(save_rdi=True, save_azimuth=False)

            elif key == ord("2"):
                save_and_update(save_rdi=False, save_azimuth=True)

            elif key == ord("3"):
                save_and_update(
                    save_rdi=False,
                    save_azimuth=False,
                    save_azimuth_unshifted=True,
                )

            elif key == ord("s"):
                save_and_update(
                    save_rdi=True,
                    save_azimuth=True,
                    save_azimuth_unshifted=True,
                )

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
