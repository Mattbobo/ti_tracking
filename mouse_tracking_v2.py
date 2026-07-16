import argparse
import math
import pickle
import sys
import time
from collections import deque

import numpy as np
import zmq
from PySide2 import QtCore, QtGui, QtWidgets

import dca1000_rdi_raw_range_azimuth_camera as dca


# === ZeroMQ Client Setup ===
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://localhost:5555")


def load_mean_std(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Invalid mean/std file: {path}")
    return float(lines[0]), float(lines[1])


def ask_angle(seq: np.ndarray):
    """Send angle request, return angle in degrees."""
    sock.send(pickle.dumps(("angle", seq), protocol=4))
    angle, _ = sock.recv_pyobj()
    return angle


def ask_dist(seq: np.ndarray):
    """Send distance request, return distance in cm."""
    sock.send(pickle.dumps(("dist", seq), protocol=4))
    _, dist = sock.recv_pyobj()
    return dist


def get_model_range_slice(args):
    """Use the same range-bin crop convention as the DCA1000 reference recorder."""
    full_range_bins = int(dca.SAVE_RANGE_BINS)
    start = args.model_range_start
    end = args.model_range_end

    if start is None:
        start = getattr(dca, "RDI_DISPLAY_RANGE_START_BIN", 0)
    if end is None:
        end = getattr(dca, "RDI_DISPLAY_RANGE_END_BIN", full_range_bins)

    start = max(0, min(int(start), full_range_bins - 1))
    end = full_range_bins if end is None else int(end)
    end = max(start + 1, min(end, full_range_bins))
    return start, end


def estimate_range_bin_cm_from_cfg(cfg_path, fft_size):
    """Estimate physical cm per Range FFT row from profileCfg."""
    if cfg_path is None or str(cfg_path).lower() == "none":
        return None

    try:
        with open(cfg_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 12 or parts[0] != "profileCfg":
                    continue

                freq_slope_mhz_us = float(parts[8])
                sample_rate_ksps = float(parts[11])
                if freq_slope_mhz_us <= 0 or sample_rate_ksps <= 0 or fft_size <= 0:
                    return None

                c_m_s = 299792458.0
                sample_rate_hz = sample_rate_ksps * 1e3
                freq_slope_hz_s = freq_slope_mhz_us * 1e12
                range_bin_m = c_m_s * sample_rate_hz / (2.0 * freq_slope_hz_s * float(fft_size))
                return range_bin_m * 100.0
    except OSError:
        return None

    return None


class DCA1000RadarReaderThread(QtCore.QThread):
    radar_frame_ready = QtCore.Signal(object)
    status_msg = QtCore.Signal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.running = False
        self.radar_ser = None
        self.receiver = None
        self.controls = dca.RuntimeRDIControls()
        self.display_range_start, self.display_range_end = get_model_range_slice(args)

    def run(self):
        self.running = True

        try:
            dca.CLI_PORT = self.args.cli
            if self.args.cfg is not None and self.args.cfg.lower() != "none":
                dca.RADAR_CFG = self.args.cfg
            dca.DCA_CONTROL_EXE = self.args.dca_control_exe
            dca.DCA_JSON = self.args.dca_json
            dca.DCA_DIR = str(dca.Path(dca.DCA_CONTROL_EXE).parent)

            self.status_msg.emit("Opening radar CLI and sending cfg...")
            self.radar_ser = dca.open_radar_cli()
            if self.args.cfg is not None and self.args.cfg.lower() != "none":
                dca.send_radar_cfg_without_sensor_start(self.radar_ser)

            self.status_msg.emit("Configuring DCA1000...")
            dca.run_dca_command("fpga", check=True)
            dca.run_dca_command("record", check=True)

            self.status_msg.emit("Starting DCA1000 UDP receiver...")
            self.receiver = dca.DCA1000UDPReceiver(controls=self.controls)
            self.receiver.start()
            self.msleep(1000)

            dca.dca_start_record_direct()
            self.msleep(200)
            dca.radar_sensor_start(self.radar_ser)
            self.status_msg.emit("DCA1000 radar running.")

            last_frame_id = -1
            while self.running:
                rdi_complex, frame_id, azimuth_map = self.receiver.get_latest_rdi()
                if rdi_complex is not None and frame_id > last_frame_id:
                    radar_frame = dca.make_two_channel_radar_frame(
                        rdi_complex,
                        azimuth_map=azimuth_map,
                        display_range_start=self.display_range_start,
                        display_range_end=self.display_range_end,
                    )
                    self.radar_frame_ready.emit({
                        "frame_number": int(frame_id),
                        "radar_frame": radar_frame.astype(np.float32),
                    })
                    last_frame_id = int(frame_id)
                self.msleep(max(1, int(self.args.poll_ms)))

        except Exception as e:
            self.status_msg.emit(f"DCA radar error: {e}")

        finally:
            if self.radar_ser is not None:
                try:
                    dca.radar_sensor_stop(self.radar_ser)
                except Exception as e:
                    self.status_msg.emit(f"Radar sensorStop warning: {e}")

            try:
                dca.dca_stop_record_direct()
            except Exception as e:
                self.status_msg.emit(f"DCA stop warning: {e}")

            if self.receiver is not None:
                self.receiver.stop()
                self.receiver = None

            if self.radar_ser is not None:
                try:
                    self.radar_ser.close()
                except Exception:
                    pass
                self.radar_ser = None

            self.running = False
            self.status_msg.emit("DCA1000 radar stopped.")

    def stop(self):
        self.running = False
        self.wait(5000)


class ShowDataView(QtWidgets.QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("TI Radar Mouse Tracking")

        self.angle_mean, self.angle_std = load_mean_std(self.args.angle_mean_std)
        self.dist_mean, self.dist_std = load_mean_std(self.args.dist_mean_std)

        self.reader_thread = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.display_range_start, self.display_range_end = get_model_range_slice(self.args)
        self.model_range_bins = self.display_range_end - self.display_range_start

        # Sequence buffer: each frame is (2, model_range_bins, 32) = [RDI, Azimuth],
        # matching dca1000_rdi_raw_range_azimuth_camera.py recording.
        self.seq_len = int(self.args.seq_len)
        self.buffer = deque(maxlen=self.seq_len)

        # Predictions in polar coordinates.
        self.pred_angle = None
        self.pred_dist = None

        # ===== Drawing/Mapping params (XY Panel) =====
        self.A_deg = 45.0
        self.Rmax_cm = 30.0
        self.Xmax_cm = self.Rmax_cm * math.sin(math.radians(self.A_deg))
        self.Ymax_cm = self.Rmax_cm

        self.PAD_Y0_CM = 12.0
        self.PAD_H_CM = 16.0
        self.PAD_W_CM_DESIRED = 20.0

        self.PAD_Y0_CM = max(0.0, min(self.PAD_Y0_CM, self.Rmax_cm))
        self.PAD_H_CM = max(1.0, min(self.PAD_H_CM, self.Rmax_cm - self.PAD_Y0_CM))

        tan_a = math.tan(math.radians(self.A_deg))
        allowed_half_w_near = self.PAD_Y0_CM * tan_a
        allowed_half_w_far = (self.PAD_Y0_CM + self.PAD_H_CM) * tan_a
        allowed_half_w = min(allowed_half_w_near, allowed_half_w_far, self.Xmax_cm)

        self.PAD_HALF_W_CM = min(self.PAD_W_CM_DESIRED / 2.0, allowed_half_w)
        self.ACT_W_CM = 2.0 * self.PAD_HALF_W_CM
        self.ACT_H_CM = self.PAD_H_CM

        # Dynamic energy gating, matching mouse_tracking_v3's idea.
        self.ENERGY_NEAR = float(
            self.args.energy_near if self.args.energy_near is not None else self.args.energy_on
        )
        self.ENERGY_FAR = float(
            self.args.energy_far if self.args.energy_far is not None else self.args.energy_off
        )
        self.ENERGY_RMAX_CM = float(self.args.energy_rmax_cm)
        self.ema_energy = None
        self.ema_alpha_energy = float(self.args.energy_ema_alpha)
        self._detected_dynamic = False

        self.range_bin_cm = self._resolve_range_bin_cm()
        self.ignore_near_cm = max(0.0, float(self.args.ignore_near_cm))
        self.ignore_near_bins = self._resolve_ignore_near_bins()

        self.point_size = 20
        self.point_target_ema_alpha = float(self.args.point_target_ema_alpha)
        self.point_smooth_alpha = float(self.args.point_smooth_alpha)
        self.max_point_step_cm = float(self.args.max_point_step_cm)
        self.point_anim_ms = max(1, int(self.args.point_anim_ms))
        self.pred_target_ema_local = None
        self.pred_target_local = None
        self.pred_display_local = None

        self.grid_cm = 2.0
        self.pred_hold_local = None
        self.pred_hold_active = False

        self.boundary_tol_cm = float(self.args.boundary_tol_cm)
        self.clear_outside_frames = int(self.args.clear_outside_frames)
        self._outside_cnt = 0

        self._build_ui()
        self.point_anim_timer = QtCore.QTimer(self)
        self.point_anim_timer.timeout.connect(self._animate_display_point)
        self.point_anim_timer.start(self.point_anim_ms)

        if self.args.auto_start:
            self.start_radar()

    def _build_ui(self):
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)

        self.pred_label = QtWidgets.QLabel("Predicted: -- deg, -- cm")
        self.pred_label.setFont(font)

        self.pred_xy_label = QtWidgets.QLabel("Pred Grid XY: --, -- (cells)")
        self.pred_xy_label.setFont(QtGui.QFont("", 12))

        self.energy_label = QtWidgets.QLabel("RDI Energy: --")
        self.energy_label.setFont(QtGui.QFont("", 12))

        self.sector_label = QtWidgets.QLabel()
        self.sector_label.setFixedSize(600, 600)

        self.legend_label = QtWidgets.QLabel("Model Prediction")
        self.legend_label.setFont(QtGui.QFont("", 12))

        self.start_btn = QtWidgets.QPushButton("Start Radar")
        self.stop_btn = QtWidgets.QPushButton("Stop Radar")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_radar)
        self.stop_btn.clicked.connect(self.stop_radar)

        layout = QtWidgets.QVBoxLayout(self)

        radar_ctrl = QtWidgets.QHBoxLayout()
        radar_ctrl.addWidget(self.start_btn)
        radar_ctrl.addWidget(self.stop_btn)
        radar_ctrl.addStretch()
        layout.addLayout(radar_ctrl)

        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.pred_label)
        hl.addStretch()
        layout.addLayout(hl)

        hl2 = QtWidgets.QHBoxLayout()
        hl2.addWidget(self.pred_xy_label)
        hl2.addStretch()
        layout.addLayout(hl2)

        hl3 = QtWidgets.QHBoxLayout()
        hl3.addWidget(self.energy_label)
        hl3.addStretch()
        layout.addLayout(hl3)

        layout.addWidget(self.sector_label, alignment=QtCore.Qt.AlignHCenter)
        layout.addWidget(self.legend_label, alignment=QtCore.Qt.AlignHCenter)

        self.updateXY()

    def start_radar(self):
        if self.reader_thread is not None:
            return

        self.reader_thread = DCA1000RadarReaderThread(self.args)
        self.reader_thread.radar_frame_ready.connect(self.updateRadar)
        self.reader_thread.status_msg.connect(self.update_radar_status)
        self.reader_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_radar(self):
        if self.reader_thread is not None:
            self.reader_thread.stop()
            self.reader_thread = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_radar_status(self, msg):
        print("[RADAR]", msg)

    def _update_fps(self):
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now

    def _extract_radar_frame(self, packet):
        arr = packet.get("radar_frame")
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32)
        expected_shape = (2, self.model_range_bins, dca.SAVE_DOPPLER_BINS)
        if arr.shape != expected_shape:
            print(f"[WARN] Unexpected DCA radar frame shape: {arr.shape}, expected {expected_shape}")
            return None
        return arr

    def _resolve_range_bin_cm(self) -> float:
        explicit = self.args.range_bin_cm
        if explicit is not None and float(explicit) > 0:
            return float(explicit)

        estimated = estimate_range_bin_cm_from_cfg(
            self.args.cfg,
            int(getattr(dca, "RANGE_FFT_SIZE", dca.SAVE_RANGE_BINS)),
        )
        if estimated is not None and estimated > 0:
            return float(estimated)

        return 1.0

    def _resolve_ignore_near_bins(self) -> int:
        if self.ignore_near_cm <= 0.0 or self.range_bin_cm <= 0.0:
            return 0
        bins = int(math.ceil(self.ignore_near_cm / self.range_bin_cm))
        return max(0, min(bins, max(0, self.model_range_bins - 1)))

    def _detection_energy(self, arr: np.ndarray) -> float:
        # Ignore only the near-side rows for gating; keep model input unchanged.
        work = np.abs(arr)
        if self.ignore_near_bins > 0:
            work = work[:, self.ignore_near_bins:, :]
        return float(np.sum(work))

    def _inside_pad(self, x_local: float, y_local: float) -> bool:
        return (-self.ACT_W_CM / 2 <= x_local <= self.ACT_W_CM / 2) and (0.0 <= y_local <= self.ACT_H_CM)

    def _inside_pad_strict(self, x_local: float, y_local: float) -> bool:
        return (-self.ACT_W_CM / 2 <= x_local <= self.ACT_W_CM / 2) and (0.0 <= y_local <= self.ACT_H_CM)

    def _inside_pad_loose(self, x_local: float, y_local: float) -> bool:
        t = self.boundary_tol_cm
        return (-self.ACT_W_CM / 2 - t <= x_local <= self.ACT_W_CM / 2 + t) and (-t <= y_local <= self.ACT_H_CM + t)

    def polar_to_xy_cm(self, angle_deg: float, dist_cm: float):
        rad = math.radians(angle_deg)
        x = dist_cm * math.sin(rad)
        y = dist_cm * math.cos(rad)
        return x, y

    def _dynamic_energy_threshold(self, dist_cm: float) -> float:
        rmax = max(1.0, self.ENERGY_RMAX_CM)
        d = max(1.0, min(rmax, float(dist_cm)))
        return self.ENERGY_NEAR - (d / rmax) * (self.ENERGY_NEAR - self.ENERGY_FAR)

    def _update_gate_dynamic(self, total_energy: float, dist_cm: float) -> bool:
        alpha = max(0.0, min(1.0, self.ema_alpha_energy))
        if self.ema_energy is None:
            self.ema_energy = total_energy
        else:
            self.ema_energy = (1.0 - alpha) * self.ema_energy + alpha * total_energy

        threshold = self._dynamic_energy_threshold(dist_cm)
        self._detected_dynamic = self.ema_energy >= threshold

        if self.args.debug:
            print(
                f"energy={total_energy:.1f}, ema={self.ema_energy:.1f}, "
                f"th={threshold:.1f}, dist={float(dist_cm):.1f}, "
                f"ignore_bins={self.ignore_near_bins}"
            )

        return self._detected_dynamic

    def _show_held_prediction(self):
        if self.pred_hold_local is None:
            return

        hx, hy = self.pred_hold_local
        self._move_display_point_toward(hx, hy)
        self.pred_hold_active = True
        self.pred_angle = None
        self.pred_dist = None
        self.pred_label.setText("Predicted: -- deg, -- cm")
        self.pred_xy_label.setText(
            f"Pred Grid XY: {hx / self.grid_cm:.1f}, {hy / self.grid_cm:.1f} (cells)"
        )
        self.updateXY()

    def _smooth_point_target(self, x_local: float, y_local: float):
        raw = np.array([float(x_local), float(y_local)], dtype=np.float32)
        alpha = max(0.0, min(1.0, self.point_target_ema_alpha))

        if self.pred_target_ema_local is None:
            self.pred_target_ema_local = raw.copy()
        else:
            self.pred_target_ema_local = (alpha * raw + (1.0 - alpha) * self.pred_target_ema_local).astype(
                np.float32
            )

        return self.pred_target_ema_local

    def _move_display_point_toward(self, x_local: float, y_local: float):
        target = np.array([float(x_local), float(y_local)], dtype=np.float32)
        self.pred_target_local = target

        if self.pred_display_local is None:
            self.pred_display_local = target.copy()
            return

        display = np.asarray(self.pred_display_local, dtype=np.float32)
        delta = target - display
        delta_len = float(np.linalg.norm(delta))
        if delta_len <= 1e-6:
            self.pred_display_local = target.copy()
            return

        alpha = max(0.0, min(1.0, self.point_smooth_alpha))
        step = delta * alpha
        step_len = float(np.linalg.norm(step))
        max_step = max(0.0, self.max_point_step_cm)

        if max_step > 0.0 and step_len > max_step:
            step = delta / delta_len * max_step

        next_display = (display + step).astype(np.float32)
        if float(np.linalg.norm(target - next_display)) < 0.03:
            next_display = target.copy()
        self.pred_display_local = next_display

    def _animate_display_point(self):
        if self.pred_target_local is None or self.pred_display_local is None:
            return
        if self.pred_angle is None and not self.pred_hold_active:
            return

        before = np.asarray(self.pred_display_local, dtype=np.float32).copy()
        tx, ty = self.pred_target_local
        self._move_display_point_toward(float(tx), float(ty))

        if float(np.linalg.norm(self.pred_display_local - before)) > 1e-4:
            self.updateXY()

    def _clear_prediction(self):
        self.pred_hold_active = False
        self.pred_hold_local = None
        self.pred_angle = None
        self.pred_dist = None
        self.pred_target_ema_local = None
        self.pred_target_local = None
        self.pred_display_local = None
        self._outside_cnt = 0
        self.pred_label.setText("Predicted: -- deg, -- cm")
        self.pred_xy_label.setText("Pred Grid XY: --, -- (cells)")
        self.updateXY()

    def updateRadar(self, packet):
        arr = self._extract_radar_frame(packet)
        if arr is None:
            return

        self._update_fps()

        total_energy = self._detection_energy(arr)
        self.energy_label.setText(f"RDI Energy: {total_energy:,.1f}")

        self.buffer.append(arr)
        if len(self.buffer) < self.seq_len:
            self._clear_prediction()
            return

        seq = np.stack(self.buffer, axis=0).astype(np.float32)
        seq_angle = (seq - self.angle_mean) / self.angle_std
        seq_dist = (seq[:, 0:1, :, :] - self.dist_mean) / self.dist_std

        angle = ask_angle(seq_angle)
        dist = ask_dist(seq_dist)
        if angle is None or dist is None:
            self._clear_prediction()
            return

        detected = self._update_gate_dynamic(total_energy, float(dist))

        gx, gy = self.polar_to_xy_cm(float(angle), float(dist))
        x_local_raw = gx
        y_local_raw = gy - self.PAD_Y0_CM

        if not detected:
            if self.pred_hold_local is not None:
                hx, hy = self.pred_hold_local
                if self._inside_pad_loose(hx, hy):
                    self._outside_cnt = 0
                    self._show_held_prediction()
                    return

                self._outside_cnt += 1
                if self._outside_cnt < self.clear_outside_frames:
                    self._show_held_prediction()
                    return

            self._clear_prediction()
            return

        self.pred_hold_active = False
        self.pred_angle, self.pred_dist = float(angle), float(dist)
        self.pred_label.setText(f"Predicted: {float(angle):.1f} deg, {float(dist):.1f} cm")

        x_u = -x_local_raw
        y_u = self.ACT_H_CM - y_local_raw

        if not self._inside_pad_loose(x_u, y_u):
            self._outside_cnt += 1
            if self._outside_cnt >= self.clear_outside_frames:
                self._clear_prediction()
            elif self.pred_hold_local is not None:
                self._show_held_prediction()
            else:
                self.updateXY()
            return

        self._outside_cnt = 0

        if not self._inside_pad_strict(x_u, y_u):
            x_local = max(-self.ACT_W_CM / 2, min(self.ACT_W_CM / 2, x_u))
            y_local = max(0.0, min(self.ACT_H_CM, y_u))
        else:
            x_local, y_local = x_u, y_u

        target_local = self._smooth_point_target(x_local, y_local)
        tx, ty = float(target_local[0]), float(target_local[1])
        self.pred_hold_local = (tx, ty)
        self._move_display_point_toward(tx, ty)
        self.pred_xy_label.setText(
            f"Pred Grid XY: {tx / self.grid_cm:.1f}, {ty / self.grid_cm:.1f} (cells)"
        )
        self.updateXY()

    def updateXY(self):
        w, h = self.sector_label.width(), self.sector_label.height()
        pix = QtGui.QPixmap(w, h)
        pix.fill(QtCore.Qt.black)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        p.setPen(QtGui.QPen(QtCore.Qt.white, 2))
        p.drawRect(0, 0, w - 1, h - 1)

        margin = 20
        cell = self.grid_cm
        nx = int(round(self.ACT_W_CM / cell))
        ny = int(round(self.ACT_H_CM / cell))

        s = min((w - 2 * margin) / nx, (h - 2 * margin) / ny)
        rect_w = int(round(nx * s))
        rect_h = int(round(ny * s))

        rect_left = int(round(w / 2 - rect_w / 2))
        rect_right = rect_left + rect_w
        rect_bottom = int(h - margin)
        rect_top = rect_bottom - rect_h

        p.setPen(QtGui.QPen(QtCore.Qt.white, 3))
        p.drawRect(QtCore.QRect(rect_left, rect_top, rect_w, rect_h))

        p.setPen(QtGui.QPen(QtCore.Qt.gray, 1))
        for i in range(1, ny):
            y = rect_bottom - int(round(i * s))
            p.drawLine(rect_left, y, rect_right, y)
        for j in range(1, nx):
            x = rect_left + int(round(j * s))
            p.drawLine(x, rect_top, x, rect_bottom)

        def local_cm_to_px(x_local: float, y_local: float):
            x_local = max(-self.ACT_W_CM / 2, min(self.ACT_W_CM / 2, x_local))
            y_local = max(0.0, min(self.ACT_H_CM, y_local))
            gx = x_local / cell
            gy = y_local / cell
            px = int(round((rect_left + rect_right) / 2 + gx * s))
            py = int(round(rect_bottom - gy * s))
            return px, py

        def draw_local_point(x_local, y_local, color):
            px, py = local_cm_to_px(float(x_local), float(y_local))
            p.setBrush(QtGui.QBrush(color))
            p.setPen(QtGui.QPen(color, 1))
            p.drawEllipse(QtCore.QPointF(px, py), self.point_size, self.point_size)

        if self.pred_display_local is not None and (self.pred_angle is not None or self.pred_hold_active):
            dx, dy = self.pred_display_local
            if self._inside_pad_loose(float(dx), float(dy)):
                draw_local_point(dx, dy, QtGui.QColor(255, 80, 80))

        p.end()
        self.sector_label.setPixmap(pix)

    def closeEvent(self, event):
        self.stop_radar()
        event.accept()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Online mouse tracking with the DCA1000 raw UDP radar pipeline."
    )
    parser.add_argument("--cli", type=str, default=dca.CLI_PORT, help="Radar CLI COM port.")
    parser.add_argument("--cfg", type=str, default=dca.RADAR_CFG, help="mmWave cfg path. Use 'none' to skip.")
    parser.add_argument("--dca-control-exe", type=str, default=dca.DCA_CONTROL_EXE)
    parser.add_argument("--dca-json", type=str, default=dca.DCA_JSON)
    parser.add_argument("--poll-ms", type=int, default=2, help="Polling interval for latest DCA1000 frame.")
    parser.add_argument(
        "--model-range-start",
        type=int,
        default=None,
        help="First range bin used for the model input. Default follows the reference RDI display crop.",
    )
    parser.add_argument(
        "--model-range-end",
        type=int,
        default=None,
        help="Exclusive range-bin end used for the model input. Default follows the reference RDI display crop.",
    )
    parser.add_argument("--angle-mean-std", type=str, default=r"Process_data\Angle_data\mean_std.txt")
    parser.add_argument("--dist-mean-std", type=str, default=r"Process_data\Distance_data\mean_std.txt")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--energy-on", type=float, default=100000.0, help="Legacy alias for --energy-near.")
    parser.add_argument("--energy-off", type=float, default=85000.0, help="Legacy alias for --energy-far.")
    parser.add_argument("--energy-near", type=float, default=None)
    parser.add_argument("--energy-far", type=float, default=None)
    parser.add_argument("--energy-rmax-cm", type=float, default=40.0)
    parser.add_argument("--energy-ema-alpha", type=float, default=0.3)
    parser.add_argument(
        "--ignore-near-cm",
        type=float,
        default=15.0,
        help="Near-side raw radar rows ignored for energy gating only; model input is unchanged.",
    )
    parser.add_argument(
        "--range-bin-cm",
        type=float,
        default=None,
        help="Override cm per active radar range row for --ignore-near-cm. Default is estimated from cfg.",
    )
    parser.add_argument("--boundary-tol-cm", type=float, default=1.0)
    parser.add_argument("--clear-outside-frames", type=int, default=8)
    parser.add_argument(
        "--point-target-ema-alpha",
        type=float,
        default=0.3,
        help="EMA alpha for model XY target before display animation. Lower values reduce jitter.",
    )
    parser.add_argument(
        "--point-smooth-alpha",
        type=float,
        default=0.45,
        help="Fraction of the remaining target distance the red point moves each radar frame.",
    )
    parser.add_argument(
        "--max-point-step-cm",
        type=float,
        default=1.2,
        help="Maximum red-point movement per animation tick in display cm. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--point-anim-ms",
        type=int,
        default=33,
        help="Red-point animation timer interval in ms.",
    )
    parser.add_argument("--auto-start", action="store_true", default=True)
    parser.add_argument("--no-auto-start", action="store_false", dest="auto_start")
    parser.add_argument("--debug", action="store_true")
    return parser


def main():
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()

    app = QtWidgets.QApplication(sys.argv)
    win = ShowDataView(args)
    win.resize(820, 900)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
