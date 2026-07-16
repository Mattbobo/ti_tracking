import argparse
import csv
import os
import struct
import sys
import time

import cv2
import h5py
import numpy as np
import serial
from PySide2 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"

MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4
MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP = 8
AZIMUTH_TLV_TYPES = (
    MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP,
    MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP,
)


def send_config(cli_port, cfg_path, baudrate=115200):
    print(f"[INFO] Opening CLI port: {cli_port}")
    cli = serial.Serial(cli_port, baudrate, timeout=1)
    time.sleep(0.1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("%"):
            continue

        print(f"[CFG] {line}")
        cli.write((line + "\n").encode())
        time.sleep(0.05)

        try:
            resp = cli.read(cli.in_waiting or 1).decode(errors="ignore")
            if resp.strip():
                print(resp.strip())
        except Exception:
            pass

    print("[INFO] Config sent.")
    cli.close()


def find_magic(buffer):
    return buffer.find(MAGIC_WORD)


def parse_packet(buffer):
    magic_idx = find_magic(buffer)
    if magic_idx < 0:
        return None, buffer[-7:]

    if magic_idx > 0:
        buffer = buffer[magic_idx:]

    if len(buffer) < 40:
        return None, buffer

    try:
        header = struct.unpack_from("<QIIIIIIII", buffer, 0)
    except struct.error:
        return None, buffer

    total_packet_len = header[2]
    if total_packet_len <= 0:
        return None, buffer[8:]

    if len(buffer) < total_packet_len:
        return None, buffer

    packet = buffer[:total_packet_len]
    remaining = buffer[total_packet_len:]

    parsed = {
        "version": header[1],
        "total_packet_len": total_packet_len,
        "platform": header[3],
        "frame_number": header[4],
        "time_cpu_cycles": header[5],
        "num_detected_obj": header[6],
        "num_tlvs": header[7],
        "sub_frame_number": header[8],
        "tlvs": [],
    }

    offset = 40
    for _ in range(parsed["num_tlvs"]):
        if offset + 8 > len(packet):
            break

        tlv_type, tlv_length = struct.unpack_from("<II", packet, offset)
        offset += 8

        payload_len = tlv_length
        if offset + payload_len > len(packet):
            if tlv_length >= 8 and offset + (tlv_length - 8) <= len(packet):
                payload_len = tlv_length - 8
            else:
                print(
                    f"[WARN] TLV payload out of packet: type={tlv_type}, "
                    f"length={tlv_length}, offset={offset}, packetLen={len(packet)}"
                )
                break

        payload = packet[offset: offset + payload_len]
        offset += payload_len

        parsed["tlvs"].append({
            "type": tlv_type,
            "length": tlv_length,
            "payload": payload,
        })

    return parsed, remaining


def extract_rdi_from_packet(packet, range_bins, doppler_bins, keep_range_m=None, max_range_m=2.41):
    expected_bytes = range_bins * doppler_bins * 2

    for tlv in packet["tlvs"]:
        if tlv["type"] != MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP:
            continue

        payload = tlv["payload"]
        if len(payload) < expected_bytes:
            print(f"[WARN] RDI payload too small: {len(payload)} bytes, expected {expected_bytes}")
            return None

        data = np.frombuffer(payload[:expected_bytes], dtype=np.uint16)
        if data.size != range_bins * doppler_bins:
            print(f"[WARN] RDI size mismatch: {data.size}, expected {range_bins * doppler_bins}")
            return None

        rdi_raw = data.reshape((range_bins, doppler_bins))
        rdi_shift = np.fft.fftshift(rdi_raw, axes=1)
        rdi_plot = rdi_shift.T
        rdi_plot_db = 20 * np.log10(rdi_plot.astype(np.float32) + 1.0)

        if keep_range_m is not None and keep_range_m > 0:
            keep_bins = int(np.ceil(range_bins * keep_range_m / max_range_m))
            keep_bins = max(1, min(keep_bins, range_bins))
            rdi_plot_db = rdi_plot_db[:, :keep_bins]

        return rdi_plot_db.astype(np.float32)

    return None


def extract_azimuth_from_packet(
    packet,
    range_bins,
    num_virtual_ant,
    angle_fft_bins=128,
    iq_order="ri",
    keep_range_m=None,
    max_range_m=2.41,
):
    bytes_per_range_ant = 4

    for tlv in packet["tlvs"]:
        if tlv["type"] not in AZIMUTH_TLV_TYPES:
            continue

        payload = tlv["payload"]
        if len(payload) < range_bins * bytes_per_range_ant:
            print(
                f"[WARN] Azimuth payload too small: type={tlv['type']}, "
                f"payload={len(payload)} bytes, range_bins={range_bins}"
            )
            return None

        inferred_ant = len(payload) // (range_bins * bytes_per_range_ant)
        if inferred_ant <= 0:
            print(f"[WARN] Cannot infer azimuth virtual antennas from payload={len(payload)}")
            return None

        if inferred_ant != num_virtual_ant:
            print(f"[INFO] Azimuth virtual antenna auto-adjust: arg={num_virtual_ant}, inferred={inferred_ant}")
            num_virtual_ant = inferred_ant

        expected_bytes = range_bins * num_virtual_ant * bytes_per_range_ant
        raw = np.frombuffer(payload[:expected_bytes], dtype=np.int16)
        if raw.size != range_bins * num_virtual_ant * 2:
            print(
                f"[WARN] Azimuth size mismatch: {raw.size}, "
                f"expected {range_bins * num_virtual_ant * 2}"
            )
            return None

        raw = raw.reshape((range_bins, num_virtual_ant, 2))
        if iq_order.lower() == "ir":
            complex_ant = raw[:, :, 1].astype(np.float32) + 1j * raw[:, :, 0].astype(np.float32)
        else:
            complex_ant = raw[:, :, 0].astype(np.float32) + 1j * raw[:, :, 1].astype(np.float32)

        az = np.fft.fft(complex_ant, n=angle_fft_bins, axis=1)
        az = np.fft.fftshift(az, axes=1)
        az_plot_db = 20 * np.log10(np.abs(az).astype(np.float32) + 1.0)
        az_plot_db = az_plot_db.T

        if keep_range_m is not None and keep_range_m > 0:
            keep_bins = int(np.ceil(range_bins * keep_range_m / max_range_m))
            keep_bins = max(1, min(keep_bins, range_bins))
            az_plot_db = az_plot_db[:, :keep_bins]

        return az_plot_db.astype(np.float32)

    return None


def smooth_2d(img):
    padded = np.pad(img, ((1, 1), (1, 1)), mode="edge")
    return (
        1 * padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + 1 * padded[:-2, 2:] +
        2 * padded[1:-1, :-2] + 4 * padded[1:-1, 1:-1] + 2 * padded[1:-1, 2:] +
        1 * padded[2:, :-2] + 2 * padded[2:, 1:-1] + 1 * padded[2:, 2:]
    ) / 16.0


def crop_range_for_display(img, max_range_m, display_range_m):
    if img is None or img.size == 0:
        return img

    if display_range_m is None or display_range_m <= 0:
        return img

    h, w = img.shape
    keep_bins = int(np.ceil(w * float(display_range_m) / float(max_range_m)))
    keep_bins = max(1, min(keep_bins, w))
    return img[:, :keep_bins]


def resize_2d_bilinear(img, out_h=128, out_w=128):
    img = np.asarray(img, dtype=np.float32)

    if img.ndim != 2 or img.size == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)

    in_h, in_w = img.shape

    if in_h == out_h and in_w == out_w:
        return img.astype(np.float32)

    if in_h == 1 and in_w == 1:
        return np.full((out_h, out_w), float(img[0, 0]), dtype=np.float32)

    y = np.linspace(0, in_h - 1, out_h)
    x = np.linspace(0, in_w - 1, out_w)

    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)

    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)

    top = img[y0[:, None], x0[None, :]] * (1.0 - wx[None, :]) + img[y0[:, None], x1[None, :]] * wx[None, :]
    bottom = img[y1[:, None], x0[None, :]] * (1.0 - wx[None, :]) + img[y1[:, None], x1[None, :]] * wx[None, :]
    out = top * (1.0 - wy[:, None]) + bottom * wy[:, None]

    return out.astype(np.float32)


def resize_2d_nearest(img, out_h=128, out_w=128):
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2 or img.size == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)

    in_h, in_w = img.shape
    y_idx = np.round(np.linspace(0, in_h - 1, out_h)).astype(np.int32)
    x_idx = np.round(np.linspace(0, in_w - 1, out_w)).astype(np.int32)
    return img[y_idx[:, None], x_idx[None, :]].astype(np.float32)


def resize_for_display(img, out_h, out_w, method="bilinear"):
    if method == "nearest":
        return resize_2d_nearest(img, out_h, out_w)
    return resize_2d_bilinear(img, out_h, out_w)


def official_style_rdi(img_db, noise_percentile=55.0, threshold_db=1.0, gain=1.8, clip_db=16.0):
    x = np.asarray(img_db, dtype=np.float32)
    floor = np.percentile(x, noise_percentile)
    x = x - floor
    x = np.maximum(x - threshold_db, 0.0)
    x = x * gain
    x = np.clip(x, 0.0, clip_db)
    return x.astype(np.float32)


def official_style_azimuth(
    img_db,
    range_median_weight=1.0,
    global_noise_percentile=55.0,
    threshold_db=0.6,
    gain=2.2,
    clip_db=14.0,
):
    x = np.asarray(img_db, dtype=np.float32)

    if x.size == 0:
        return x

    if range_median_weight > 0:
        range_bg = np.median(x, axis=0, keepdims=True)
        x = x - range_median_weight * range_bg

    floor = np.percentile(x, global_noise_percentile)
    x = x - floor
    x = np.maximum(x - threshold_db, 0.0)
    x = x * gain
    x = np.clip(x, 0.0, clip_db)

    return x.astype(np.float32)


class RadarReaderThread(QtCore.QThread):
    rdi_ready = QtCore.Signal(np.ndarray, int)
    azimuth_ready = QtCore.Signal(np.ndarray, int)
    radar_frame_ready = QtCore.Signal(object)
    status_msg = QtCore.Signal(str)

    def __init__(
        self,
        data_port,
        range_bins,
        doppler_bins,
        num_virtual_ant,
        angle_fft_bins,
        az_iq_order,
        keep_range_m=None,
        max_range_m=2.41,
        baudrate=921600,
        debug=False,
    ):
        super().__init__()
        self.data_port = data_port
        self.range_bins = range_bins
        self.doppler_bins = doppler_bins
        self.num_virtual_ant = num_virtual_ant
        self.angle_fft_bins = angle_fft_bins
        self.az_iq_order = az_iq_order
        self.keep_range_m = keep_range_m
        self.max_range_m = max_range_m
        self.baudrate = baudrate
        self.debug = debug
        self.running = False
        self.ser = None
        self.buffer = b""

    def run(self):
        self.running = True

        try:
            self.status_msg.emit(f"Opening data port: {self.data_port}")
            self.ser = serial.Serial(self.data_port, self.baudrate, timeout=0.01)
            self.status_msg.emit("Radar data port opened.")
        except Exception as e:
            self.status_msg.emit(f"Failed to open radar data port: {e}")
            return

        while self.running:
            try:
                n = self.ser.in_waiting
                if n > 0:
                    self.buffer += self.ser.read(n)

                packet, self.buffer = parse_packet(self.buffer)
                if packet is None:
                    self.msleep(1)
                    continue

                if self.debug:
                    print(
                        f"[FRAME] frame={packet['frame_number']}, "
                        f"numTLVs={packet['num_tlvs']}, packetLen={packet['total_packet_len']}"
                    )
                    for tlv in packet["tlvs"]:
                        print(f"    TLV type={tlv['type']}, length={tlv['length']}, payload={len(tlv['payload'])}")

                rdi_plot_db = extract_rdi_from_packet(
                    packet,
                    self.range_bins,
                    self.doppler_bins,
                    keep_range_m=self.keep_range_m,
                    max_range_m=self.max_range_m,
                )
                if rdi_plot_db is not None:
                    self.rdi_ready.emit(rdi_plot_db, packet["frame_number"])

                azimuth_plot_db = extract_azimuth_from_packet(
                    packet,
                    self.range_bins,
                    self.num_virtual_ant,
                    self.angle_fft_bins,
                    self.az_iq_order,
                    keep_range_m=self.keep_range_m,
                    max_range_m=self.max_range_m,
                )
                if azimuth_plot_db is not None:
                    self.azimuth_ready.emit(azimuth_plot_db, packet["frame_number"])

                if rdi_plot_db is not None:
                    self.radar_frame_ready.emit({
                        "frame_number": packet["frame_number"],
                        "rdi": rdi_plot_db,
                        "azimuth": azimuth_plot_db,
                    })

            except Exception as e:
                self.status_msg.emit(f"Radar read error: {e}")
                self.msleep(10)

        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass

        self.status_msg.emit("Radar reader stopped.")

    def stop(self):
        self.running = False
        self.wait(1000)


class ShowDataView(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("TI Radar & Camera Recorder (Angle & Distance)")

        calib = np.load("calib.npz")
        self.K = calib["K"]
        self.dist = calib["dist"]
        self.fx = self.K[0, 0]
        self.cx = self.K[0, 2]

        self.align_px_thresh = 5
        self.baseline_v_thresh = 5
        self.markerLength = 0.03

        self.angles = []
        self.distances = []
        self.current_angle = np.nan
        self.current_distance = np.nan

        if self.args.keep_range_m is None:
            self.display_range_m = self.args.max_range_m
        else:
            self.display_range_m = min(float(self.args.keep_range_m), float(self.args.max_range_m))

        self.display_range_bins = int(np.ceil(
            self.args.range_bins * self.display_range_m / self.args.max_range_m
        ))
        self.display_range_bins = max(1, min(self.display_range_bins, self.args.range_bins))
        self.rdi_display_range_m = self.display_range_m
        self.az_display_range_m = self.display_range_m
        self.rdi_display_range_bins = self.display_range_bins
        self.az_display_range_bins = self.display_range_bins
        self.display_size = max(32, int(self.args.display_size))

        self.reader_thread = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.rdi_clim_low = None
        self.rdi_clim_high = None
        self.az_clim_low = None
        self.az_clim_high = None
        self.last_rdi = None
        self.last_azimuth = None
        self.last_radar = None
        self.latest_camera_frame = None

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.desired_fps = 60.0
        self.record_video_fps = float(self.args.record_video_fps)
        self.frame_interval = 1.0 / self.desired_fps

        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dict, self.params)

        self.is_recording = False
        self.total_frames = 0
        self.recorded_count = 0
        self.video_writer = None
        self.h5file = None
        self.h5ds = None

        self.setup()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(int(self.frame_interval * 1000))

        if self.args.auto_start:
            self.start_radar()

    def setup(self):
        w_view, h_view = 960, 540
        self.wg = QtWidgets.QWidget()
        self.setCentralWidget(self.wg)
        main_l = QtWidgets.QVBoxLayout(self.wg)

        radar_ctrl = QtWidgets.QHBoxLayout()
        self.radar_start_btn = QtWidgets.QPushButton("Start Radar")
        self.radar_stop_btn = QtWidgets.QPushButton("Stop Radar")
        self.radar_stop_btn.setEnabled(False)
        self.radar_start_btn.clicked.connect(self.start_radar)
        self.radar_stop_btn.clicked.connect(self.stop_radar)
        self.radar_status_label = QtWidgets.QLabel("Radar ready")
        self.radar_info_label = QtWidgets.QLabel(
            f"Data: {self.args.data} | "
            f"Raw bins: R{self.args.range_bins} x D{self.args.doppler_bins} | "
            f"Display: {self.display_size}x{self.display_size} | "
            f"Keep: {self.display_range_m:.2f} m ({self.display_range_bins} bins)"
        )
        radar_ctrl.addWidget(self.radar_start_btn)
        radar_ctrl.addWidget(self.radar_stop_btn)
        radar_ctrl.addWidget(self.radar_status_label)
        radar_ctrl.addStretch()
        radar_ctrl.addWidget(self.radar_info_label)
        main_l.addLayout(radar_ctrl)

        self.fig = Figure(figsize=(10, 4), facecolor="white")
        self.canvas = FigureCanvas(self.fig)
        self.ax_rdi = self.fig.add_subplot(121)
        self.ax_az = self.fig.add_subplot(122)

        init_img = np.zeros((self.display_size, self.display_size), dtype=np.float32)
        self.im_rdi = self.ax_rdi.imshow(
            init_img,
            aspect="auto",
            origin="lower",
            cmap="jet",
            interpolation="nearest",
            extent=[0, self.display_range_m, -self.args.max_doppler_mps, self.args.max_doppler_mps],
            vmin=0,
            vmax=self.args.rdi_clip_db,
        )
        self.fig.colorbar(self.im_rdi, ax=self.ax_rdi).set_label("Relative magnitude (dB)")
        self.ax_rdi.set_xlim(0, self.display_range_m)
        self.ax_rdi.set_xlabel("Range (meters)")
        self.ax_rdi.set_ylabel("Doppler (m/s)")
        self.title_rdi = self.ax_rdi.set_title("Live RDI - official-like display")
        self.ax_rdi.axhline(0, linestyle="--", linewidth=1, color="c")

        self.im_az = self.ax_az.imshow(
            init_img,
            aspect="auto",
            origin="lower",
            cmap="jet",
            interpolation="nearest",
            extent=[0, self.display_range_m, -self.args.max_angle_deg, self.args.max_angle_deg],
            vmin=0,
            vmax=self.args.az_clip_db,
        )
        self.fig.colorbar(self.im_az, ax=self.ax_az).set_label("Relative magnitude (dB)")
        self.ax_az.set_xlim(0, self.display_range_m)
        self.ax_az.set_xlabel("Range (meters)")
        self.ax_az.set_ylabel("Angle (degrees, display scale)")
        self.title_az = self.ax_az.set_title("Live Azimuth - official-like display")
        self.ax_az.axhline(0, linestyle="--", linewidth=1, color="c")

        self.fig.tight_layout()
        main_l.addWidget(self.canvas)

        cam_block = QtWidgets.QVBoxLayout()
        self.angle_label = QtWidgets.QLabel("Angle: -- deg")
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.angle_label.setFont(font)
        self.angle_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        cam_block.addWidget(self.angle_label, alignment=QtCore.Qt.AlignLeft)

        self.distance_label = QtWidgets.QLabel("Distance: -- cm")
        dist_font = QtGui.QFont()
        dist_font.setPointSize(16)
        dist_font.setBold(True)
        self.distance_label.setFont(dist_font)
        self.distance_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        cam_block.addWidget(self.distance_label, alignment=QtCore.Qt.AlignLeft)

        self.cam_label = QtWidgets.QLabel()
        self.cam_label.setFixedSize(w_view, h_view)
        self.cam_label.setScaledContents(True)
        cam_block.addWidget(self.cam_label)
        main_l.addLayout(cam_block)

        ctrl = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel("Radar frames to record:")
        lbl.setStyleSheet("font-size:16pt;")
        ctrl.addWidget(lbl)
        self.entry = QtWidgets.QLineEdit()
        self.entry.setFixedWidth(80)
        self.entry.setStyleSheet("font-size:16pt;")
        ctrl.addWidget(self.entry)
        ctrl.addStretch()
        self.start_btn = QtWidgets.QPushButton("Start Recording")
        self.start_btn.setStyleSheet("font-size:16pt;")
        self.start_btn.clicked.connect(self.startRecording)
        ctrl.addWidget(self.start_btn)
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("font-size:16pt;")
        ctrl.addWidget(self.status_label)
        main_l.addLayout(ctrl)
        self.resize(1200, 1250)

    def start_radar(self):
        if self.reader_thread is not None:
            return

        self.radar_status_label.setText("Starting radar...")
        self.reader_thread = RadarReaderThread(
            data_port=self.args.data,
            range_bins=self.args.range_bins,
            doppler_bins=self.args.doppler_bins,
            num_virtual_ant=self.args.num_virtual_ant,
            angle_fft_bins=self.args.angle_fft_bins,
            az_iq_order=self.args.az_iq_order,
            keep_range_m=None,
            max_range_m=self.args.max_range_m,
            baudrate=self.args.data_baud,
            debug=self.args.debug,
        )
        self.reader_thread.rdi_ready.connect(self.update_rdi)
        self.reader_thread.azimuth_ready.connect(self.update_azimuth)
        self.reader_thread.radar_frame_ready.connect(self.record_radar_frame)
        self.reader_thread.status_msg.connect(self.update_radar_status)
        self.reader_thread.start()

        self.radar_start_btn.setEnabled(False)
        self.radar_stop_btn.setEnabled(True)

    def stop_radar(self):
        if self.reader_thread is not None:
            self.reader_thread.stop()
            self.reader_thread = None

        self.radar_start_btn.setEnabled(True)
        self.radar_stop_btn.setEnabled(False)

    def update_radar_status(self, msg):
        self.radar_status_label.setText(msg)
        print("[RADAR]", msg)

    def update_rdi(self, rdi_plot_db, frame_number):
        show_img = self._prepare_rdi_image(rdi_plot_db)
        self.last_rdi = show_img
        self._refresh_last_radar()

        self.im_rdi.set_data(show_img)
        self.im_rdi.set_clim(0, self.args.rdi_clip_db)
        self._update_fps(frame_number)
        self.title_rdi.set_text(
            f"RDI Official-like - Frame {frame_number} | FPS: {self.current_fps:.1f} | "
            f"raw {rdi_plot_db.shape[0]}x{rdi_plot_db.shape[1]} -> {self.display_size}x{self.display_size}"
        )
        self.canvas.draw_idle()

    def update_azimuth(self, azimuth_plot_db, frame_number):
        show_img = self._prepare_azimuth_image(azimuth_plot_db)
        self.last_azimuth = show_img
        self._refresh_last_radar()

        self.im_az.set_data(show_img)
        self.im_az.set_clim(0, self.args.az_clip_db)
        self.title_az.set_text(
            f"Azimuth Official-like - Frame {frame_number} | "
            f"FFT {self.args.angle_fft_bins} -> {self.display_size}x{self.display_size}"
        )
        self.canvas.draw_idle()

    def _prepare_rdi_image(self, rdi_plot_db):
        img = crop_range_for_display(
            rdi_plot_db,
            max_range_m=self.args.max_range_m,
            display_range_m=self.rdi_display_range_m,
        )
        img = official_style_rdi(
            img,
            noise_percentile=self.args.rdi_noise_percentile,
            threshold_db=self.args.rdi_threshold_db,
            gain=self.args.rdi_gain,
            clip_db=self.args.rdi_clip_db,
        )
        return resize_for_display(
            img,
            out_h=self.display_size,
            out_w=self.display_size,
            method=self.args.resize_method,
        )

    def _prepare_azimuth_image(self, azimuth_plot_db):
        if azimuth_plot_db is None:
            return np.zeros((self.display_size, self.display_size), dtype=np.float32)

        img = crop_range_for_display(
            azimuth_plot_db,
            max_range_m=self.args.max_range_m,
            display_range_m=self.az_display_range_m,
        )
        img = official_style_azimuth(
            img,
            range_median_weight=self.args.az_range_median_weight,
            global_noise_percentile=self.args.az_noise_percentile,
            threshold_db=self.args.az_threshold_db,
            gain=self.args.az_gain,
            clip_db=self.args.az_clip_db,
        )
        return resize_for_display(
            img,
            out_h=self.display_size,
            out_w=self.display_size,
            method=self.args.resize_method,
        )

    def _ema_clim(self, old_low, old_high, new_low, new_high):
        if old_low is None or old_high is None:
            low, high = float(new_low), float(new_high)
        else:
            alpha = self.args.color_ema_alpha
            low = (1 - alpha) * old_low + alpha * float(new_low)
            high = (1 - alpha) * old_high + alpha * float(new_high)

        if high <= low:
            high = low + 1.0
        return low, high

    def _update_fps(self, frame_number):
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now

    def _refresh_last_radar(self):
        if self.last_rdi is None:
            return

        self.last_radar = self._compose_radar_frame(self.last_rdi, self.last_azimuth)

    def _compose_radar_frame(self, rdi, azimuth):
        if rdi is None:
            return None

        rdi = rdi.astype(np.float32)
        target_h, target_w = rdi.shape
        if azimuth is None:
            az = np.zeros((target_h, target_w), dtype=np.float32)
        else:
            az = cv2.resize(
                azimuth.astype(np.float32),
                (target_w, target_h),
                interpolation=cv2.INTER_AREA,
            )

        return np.stack([rdi, az.astype(np.float32)], axis=0)

    def record_radar_frame(self, packet):
        if not self.is_recording or self.recorded_count >= self.total_frames:
            return

        if packet.get("rdi") is None:
            return

        rdi_img = self._prepare_rdi_image(packet.get("rdi"))
        azimuth_img = self._prepare_azimuth_image(packet.get("azimuth"))
        radar_frame = self._compose_radar_frame(rdi_img, azimuth_img)
        if radar_frame is None:
            return

        if self.latest_camera_frame is not None and self.video_writer is not None:
            self.video_writer.write(self.latest_camera_frame)

        self.h5ds[self.recorded_count, ...] = radar_frame
        self.angles.append(self.current_angle)
        self.distances.append(self.current_distance)
        self.recorded_count += 1
        self.status_label.setText(
            f"Recording radar {self.recorded_count}/{self.total_frames} "
            f"(frame {packet.get('frame_number')})"
        )

        if self.recorded_count >= self.total_frames:
            self.finishRecording()

    def updateFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.latest_camera_frame = frame.copy()
        raw = frame.copy()
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)
        self.current_angle = np.nan
        self.current_distance = np.nan
        aligned = False

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(raw, corners, ids)
            tvecs = {}
            rvec0 = None

            for corners_i, tid in zip(corners, ids.flatten()):
                if tid in (0, 1, 3):
                    rvecs, tvec_array, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners_i],
                        self.markerLength,
                        self.K,
                        self.dist,
                    )
                    tvecs[tid] = tvec_array[0, 0]
                    if tid == 0:
                        rvec0 = rvecs[0, 0]

            if rvec0 is not None and {0, 1, 3}.issubset(tvecs):
                centers = {
                    tid: c.reshape(-1, 2).mean(axis=0)
                    for c, tid in zip(corners, ids.flatten()) if tid in (0, 1)
                }
                u0, v0 = centers[0]
                u1, v1 = centers[1]
                if (abs(u0 - self.cx) < self.align_px_thresh and
                        abs(v1 - v0) < self.baseline_v_thresh):
                    aligned = True

                R_tag2cam, _ = cv2.Rodrigues(rvec0)
                tag_normal = R_tag2cam[:, 2]

                base = tvecs[1] - tvecs[0]
                base_proj = base - (base.dot(tag_normal)) * tag_normal
                base_norm = base_proj / np.linalg.norm(base_proj)

                forward = np.cross(tag_normal, base_norm)
                forward = forward / np.linalg.norm(forward)

                vec03 = tvecs[3] - tvecs[0]
                proj = vec03 - (vec03.dot(tag_normal)) * tag_normal

                dist_m = np.linalg.norm(proj)
                self.current_distance = round(dist_m * 100, 1)

                x = proj.dot(base_norm)
                y = proj.dot(forward)
                angle_rad = np.arctan2(x, y)
                self.current_angle = round(np.degrees(angle_rad), 1)

        angle_text = "-- deg" if np.isnan(self.current_angle) else f"{self.current_angle:.1f} deg"
        dist_text = "-- cm" if np.isnan(self.current_distance) else f"{self.current_distance:.1f} cm"
        self.angle_label.setText(f"Angle: {angle_text}" + ("  [Aligned]" if aligned else ""))
        self.distance_label.setText(f"Distance: {dist_text}")

        disp = cv2.flip(raw, 1)
        dh, dw = disp.shape[:2]
        cv2.line(disp, (dw // 2, 0), (dw // 2, dh), (255, 255, 255), 1)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, dw, dh, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).transformed(QtGui.QTransform().scale(-1, 1))
        self.cam_label.setPixmap(pix)

    def startRecording(self):
        try:
            n = int(self.entry.text())
            assert n > 0
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", "Please enter a positive frame count.")
            return

        if self.last_radar is None:
            QtWidgets.QMessageBox.warning(self, "Radar not ready", "Please wait until the first TI radar frame arrives.")
            return
        if self.latest_camera_frame is None:
            QtWidgets.QMessageBox.warning(self, "Camera not ready", "Please wait until the first camera frame arrives.")
            return

        for i in range(3, 0, -1):
            self.status_label.setText(f"Recording starts in {i}")
            QtWidgets.QApplication.processEvents()
            time.sleep(1)

        self.angles = []
        self.distances = []
        self.total_frames = n
        self.recorded_count = 0
        self.is_recording = True
        self.start_btn.setEnabled(False)

        ts = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join("Record", f"angle_dist_record_{ts}")
        os.makedirs(base, exist_ok=True)
        self.h5file = h5py.File(os.path.join(base, f"data_{ts}.h5"), "w")

        c, h, w = self.last_radar.shape
        self.h5ds = self.h5file.create_dataset("DS1", (n, c, h, w), dtype=np.float32)
        self.video_writer = cv2.VideoWriter(
            os.path.join(base, f"video_{ts}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.record_video_fps,
            (1280, 720),
        )
        self.status_label.setText("Recording started (radar-triggered)")

    def finishRecording(self):
        self.is_recording = False
        self.start_btn.setEnabled(True)

        frames = np.arange(len(self.angles))
        ang_arr = np.array(self.angles, dtype=np.float32)
        dist_arr = np.array(self.distances, dtype=np.float32)
        for arr in (ang_arr, dist_arr):
            mask = np.isnan(arr)
            if np.any(mask) and np.any(~mask):
                arr[mask] = np.interp(frames[mask], frames[~mask], arr[~mask])

        csv_path = os.path.join(os.path.dirname(self.h5file.filename), "records.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "angle_deg", "distance_cm"])
            for i, (a, d) in enumerate(zip(ang_arr, dist_arr)):
                w.writerow([i, f"{a:.1f}", f"{d:.1f}"])

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.h5file:
            self.h5file.close()
            self.h5file = None
            self.h5ds = None

        self.status_label.setText("Done Recording")
        QtWidgets.QMessageBox.information(self, "Info", f"Recording saved\nCSV: {csv_path}")

    def closeEvent(self, event):
        self.stop_radar()
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        if self.h5file:
            self.h5file.close()
        event.accept()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="TI radar + camera recorder for angle/distance labels."
    )
    parser.add_argument("--cli", type=str, default="COM4", help="CLI COM port, e.g. COM4")
    parser.add_argument("--data", type=str, default="COM5", help="Data COM port, e.g. COM5")
    parser.add_argument("--cfg", type=str, default="2.4m_test.cfg", help="mmWave cfg path. Use 'none' to skip.")
    parser.add_argument("--range-bins", type=int, default=64)
    parser.add_argument("--doppler-bins", type=int, default=32)
    parser.add_argument("--num-virtual-ant", type=int, default=2)
    parser.add_argument("--angle-fft-bins", type=int, default=128)
    parser.add_argument("--az-iq-order", type=str, default="ri", choices=["ri", "ir"])
    parser.add_argument("--cli-baud", type=int, default=115200)
    parser.add_argument("--data-baud", type=int, default=921600)
    parser.add_argument("--max-range-m", type=float, default=2.41)
    parser.add_argument("--max-doppler-mps", type=float, default=1.0)
    parser.add_argument("--max-angle-deg", type=float, default=45.0)
    parser.add_argument("--keep-range-m", type=float, default=0.4)
    parser.add_argument(
        "--display-size",
        type=int,
        default=128,
        help="Post-process display/recording size. RDI and Azimuth are resized to display_size x display_size.",
    )
    parser.add_argument(
        "--resize-method",
        type=str,
        default="bilinear",
        choices=["bilinear", "nearest"],
        help="Display/recording resize method.",
    )
    parser.add_argument("--rdi-noise-percentile", type=float, default=55.0)
    parser.add_argument("--rdi-threshold-db", type=float, default=1.0)
    parser.add_argument("--rdi-gain", type=float, default=1.8)
    parser.add_argument("--rdi-clip-db", type=float, default=16.0)
    parser.add_argument("--az-range-median-weight", type=float, default=0.8)
    parser.add_argument("--az-noise-percentile", type=float, default=55.0)
    parser.add_argument("--az-threshold-db", type=float, default=0.6)
    parser.add_argument("--az-gain", type=float, default=2.2)
    parser.add_argument("--az-clip-db", type=float, default=14.0)
    parser.add_argument(
        "--record-video-fps",
        type=float,
        default=6.0,
        help="FPS used for saved mp4. This should match the radar frame rate.",
    )
    parser.add_argument("--smooth", action="store_true", default=True)
    parser.add_argument("--no-smooth", action="store_false", dest="smooth")
    parser.add_argument("--color-low-percentile", type=float, default=2.0)
    parser.add_argument("--color-high-percentile", type=float, default=99.5)
    parser.add_argument("--az-color-low-percentile", type=float, default=2.0)
    parser.add_argument("--az-color-high-percentile", type=float, default=99.5)
    parser.add_argument("--color-ema-alpha", type=float, default=0.15)
    parser.add_argument("--auto-start", action="store_true", default=True)
    parser.add_argument("--no-auto-start", action="store_false", dest="auto_start")
    parser.add_argument("--debug", action="store_true")
    return parser


def main():
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()

    if args.keep_range_m is not None and args.keep_range_m <= 0:
        args.keep_range_m = None

    if args.cfg is not None and args.cfg.lower() != "none":
        send_config(args.cli, args.cfg, args.cli_baud)
        time.sleep(0.1)

    app = QtWidgets.QApplication(sys.argv)
    win = ShowDataView(args)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
