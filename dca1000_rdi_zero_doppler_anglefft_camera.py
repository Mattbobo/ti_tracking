import argparse
import os
import csv
import socket
import struct
import time
import subprocess
import threading
from pathlib import Path

import serial
import h5py
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


# ============================================================
# Paths
# ============================================================

DCA_CONTROL_EXE = r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\DCA1000EVM_CLI_Control.exe"
DCA_JSON = r"C:\Users\mc2\Desktop\ti_tracking\dca1000_config.json"
RADAR_CFG = r"C:\Users\mc2\Desktop\ti_tracking\xwr68xx_lvds_continuous.cfg"

DCA_DIR = str(Path(DCA_CONTROL_EXE).parent)


# ============================================================
# Radar CLI settings
# ============================================================

CLI_PORT = "COM4"
CLI_BAUD = 115200
CMD_TIMEOUT_SEC = 3.0


# ============================================================
# DCA1000 Ethernet settings
# ============================================================

PC_IP = "192.168.33.30"
DCA_IP = "192.168.33.180"

DCA_CMD_PORT = 4096
DATA_PORT = 4098

DCA_PACKET_HEADER_BYTES = 10


# ============================================================
# DCA1000 direct UDP command protocol
# ============================================================

DCA_HEADER = 0xA55A
DCA_FOOTER = 0xEEAA

CMD_RECORD_START = 0x05
CMD_RECORD_STOP = 0x06


# ============================================================
# Radar raw data settings
# 依照目前 xwr68xx_lvds_continuous.cfg：
#   channelCfg 15 1 0      -> 4 RX
#   profileCfg ... 32 ...  -> 32 ADC samples
#   frameCfg ... 32 ...    -> 32 chirp loops
# ============================================================

NUM_CHIRPS_PER_FRAME = 32
NUM_RX = 2
NUM_ADC_SAMPLES = 32

FRAME_BYTES = NUM_CHIRPS_PER_FRAME * NUM_RX * NUM_ADC_SAMPLES * 2 * 2
# 32 * 4 * 32 * IQ(2) * int16(2) = 16,384 bytes

# Range FFT zero-padding settings.
# NUM_ADC_SAMPLES=32, RANGE_FFT_PAD_FACTOR=2 -> RANGE_FFT_SIZE=64.
# RDI 計算會保留 64 個 range FFT bins，GUI 再只顯示 bin 32~63。
RANGE_FFT_PAD_FACTOR = 2
RANGE_FFT_SIZE = NUM_ADC_SAMPLES * RANGE_FFT_PAD_FACTOR

SAVE_RANGE_BINS = RANGE_FFT_SIZE  # 2x padded full range FFT bins: 64
SAVE_DOPPLER_BINS = NUM_CHIRPS_PER_FRAME

DISPLAY_EVERY_N_FRAMES = 50

SAVE_LAST_MODEL_INPUT = True
LAST_MODEL_INPUT_PATH = Path("last_model_input_8x64x32_AIC_range_pad2.npy")


# ============================================================
# Realtime RDI visualization settings
# ============================================================
# 顯示的是「AIC 後、Range FFT + Doppler FFT 後」的 per-RX complex RDI magnitude。
# complex RDI 本身不能直接畫成熱圖，因此這裡預設顯示 20log10(|RDI|)。
SHOW_RDI_REALTIME = True
RDI_DISPLAY_INTERVAL_SEC = 0.05      # GUI 更新間隔，0.05 秒約等於最高 20 FPS
RDI_DISPLAY_DYNAMIC_RANGE_DB = 45.0  # 若 RDI_USE_FIXED_DB_SCALE=False，才會使用這個相對動態範圍

# 固定絕對 dB 顯示範圍。
# 注意：這裡的 dB 是 20log10(abs(FFT output))，是程式內部的絕對數值，
# 不是已校正過的 dBm。
RDI_USE_FIXED_DB_SCALE = True
# RDI_FIXED_VMIN_DB = 35.0
# RDI_FIXED_VMAX_DB = 95.0
RDI_FIXED_VMIN_DB = 80.0
RDI_FIXED_VMAX_DB = 110.0

# 是否把低於門檻的值壓成背景色。
# 如果想看到完整底噪，就設 False。
RDI_MASK_LOW_DB = False
RDI_MASK_THRESHOLD_DB = 45.0

# ============================================================
# Background-normalized display settings
# ============================================================
# 這個模式只影響「GUI 顯示」，不改變 last_model_input 與模型輸入。
# 啟動後，前 RDI_BACKGROUND_WARMUP_FRAMES 幀會用來建立背景 magnitude。
# 顯示值為：20log10((current_mag + eps) / (background_mag + eps))。
# 因此沒有物體時會接近 0 dB，畫面應該接近深藍；物體進來時才會變亮。
# 舊版 magnitude-ratio 背景差分顯示。
# 若 RDI_USE_COMPLEX_BG_SUBTRACTION_DISPLAY=True，會優先使用 complex 背景扣除顯示，
# 這個 magnitude-ratio 模式不會生效。
RDI_USE_BACKGROUND_DELTA_DISPLAY = False
RDI_BACKGROUND_WARMUP_FRAMES = 100
RDI_DELTA_VMIN_DB = 0.0
RDI_DELTA_VMAX_DB = 80.0
RDI_CLIP_NEGATIVE_DELTA_DB = True

# ============================================================
# Complex-RDI background subtraction display settings
# ============================================================
# 這個模式只影響 GUI 顯示，不改變 last_model_input 與模型輸入。
# 啟動後，前 RDI_COMPLEX_BG_WARMUP_FRAMES 幀會用來建立 complex RDI 背景：
#     bg_complex = mean(rdi_complex over warmup frames)
# 顯示時使用：
#     residual = rdi_complex - bg_complex
#     display_db = 20log10(abs(residual) + eps)
# 因此仍然是先做 complex subtraction，再取 magnitude 顯示。
RDI_USE_COMPLEX_BG_SUBTRACTION_DISPLAY = False
RDI_COMPLEX_BG_WARMUP_FRAMES = 100
RDI_COMPLEX_BG_VMIN_DB = 60.0
RDI_COMPLEX_BG_VMAX_DB = 120.0

# 按 b 可以重新校正背景。
RDI_RESET_BACKGROUND_KEY = ord('b')

# ============================================================
# OpenCV integrated runtime control settings
# ============================================================
# 控制項會直接畫在 RDI 主視窗下方，不再另外開 RDI Controls 視窗。
# AIC 會影響實際 RDI 計算；complex background subtraction 只影響 GUI 顯示。
RDI_SLIDER_DB_MIN = 0
RDI_SLIDER_DB_MAX = 180
RDI_INTEGRATED_WINDOW_MIN_WIDTH = 1180
RDI_INTEGRATED_CONTROL_HEIGHT = 380

# ============================================================
# Static clutter / zero-Doppler suppression settings
# ============================================================
# 這個會改變 RDI 計算本身：在 Range FFT 後、Doppler FFT 前，
# 對每個 (range, rx) 沿 chirp/slow-time 維度扣掉平均值。
# 可有效壓掉 Doppler 中心 d=NUM_CHIRPS_PER_FRAME//2 的靜態亮線。
# 注意：如果手停住或移動很慢，也會被一起削弱；若要觀察完整原始 RDI，改成 False。
STATIC_CLUTTER_REMOVAL = False

# 在每張 RX 圖上標出目前最強 peak 的 range/doppler bin。
RDI_DRAW_PEAK_MARKER = True
RDI_PEAK_IGNORE_CENTER_DOPPLER = False
RDI_PEAK_CENTER_HALF_WIDTH = 1  # 若忽略中心 Doppler，會遮掉 center±這個寬度

RDI_DISPLAY_EPS = 1e-6
RDI_DISPLAY_ORIGIN = "lower"   # OpenCV 顯示時會用 np.flipud 模擬 lower origin

# ============================================================
# GUI range-bin display crop settings
# ============================================================
# 只影響 GUI 顯示，不改變 RDI 計算、不改變 last_model_input 儲存。
# mode:
#   "full"             : 顯示全部 range bins
#   "upper_half_screen": 顯示畫面上方一半的 range bins
#   "lower_half_screen": 顯示畫面下方一半的 range bins
#   "custom"           : 使用 RDI_DISPLAY_RANGE_START_BIN / END_BIN
#
# 目前 RANGE_FFT_SIZE=64，這裡指定 GUI 只顯示畫面上方 1/3 的 range bins。
# RDI_DISPLAY_ORIGIN="lower" 時，畫面上方代表較大的 range bin，
# 因此實際顯示最後約 1/3 的 range bins：bin 42~63。
RDI_DISPLAY_RANGE_MODE = "custom"
RDI_DISPLAY_RANGE_START_BIN = RANGE_FFT_SIZE // 2
RDI_DISPLAY_RANGE_END_BIN = RANGE_FFT_SIZE

# OpenCV 顯示視窗大小。GUI 目前只顯示上方 1/3 range bins，共約 22x32，這裡放大顯示。
RDI_TILE_WIDTH = 240
RDI_TILE_HEIGHT = 240

# Range distance visualization.
# 原本 RDI 只有 16 個 range bins，很難用肉眼看距離，
# 所以這版會在每張 RDI 右側加上 1D range profile，
# 並在 RDI 圖上畫出目前 range peak 的水平線。
RDI_DRAW_RANGE_LINE = False
RDI_DRAW_RANGE_PROFILE = False
RDI_PROFILE_WIDTH = 100
RDI_PROFILE_GAP = 10
RDI_RANGE_PROFILE_MODE = "max"  # "max" / "mean"
RDI_RANGE_IGNORE_CENTER_DOPPLER_FOR_PROFILE = False
RDI_RANGE_CENTER_HALF_WIDTH = 1
RDI_IGNORE_NEAR_RANGE_BINS = 0  # 若 range bin 0 leakage 太強，可先設 1 或 2

# 2x2 RX tiles 的間距與標題列高度。
# GAP 越大，四張圖越分開；BORDER 是外框留白。
RDI_TILE_GAP = 18
RDI_TILE_BORDER = 12
RDI_TITLE_BAR_HEIGHT = 38
RDI_WINDOW_NAME = "Realtime RDI + Zero-Doppler Azimuth + AprilTag Camera"


# ============================================================
# Azimuth/PHD map settings
# ============================================================
# 右邊 Azimuth 圖改成「近 0 速 Doppler bins 的 antenna FFT」。
#
# 原本固定取 zero-Doppler bin，容易取到靜態背景 / DC leakage，
# 拿去訓練角度模型時會讓模型幾乎學不到左右角度。
#
# 新流程：
#   1. 取 Doppler FFT 後中心附近的 low-speed bins。
#   2. 排除真正 zero-Doppler center bin。
#   3. 對 near-zero bins 做 complex weighted sum，保留 RX 間相位差。
#   4. 對 RX1/RX2 這個 antenna 維度做 Angle FFT。
#   5. Angle FFT bins 設成跟 Doppler bins 一樣，所以輸出維度會跟 RDI 一樣：
#      range_bins x doppler_bins。
#
# 以 doppler_bins=32 為例，center=16，half_width=2：
#   使用 bins = [14, 15, 17, 18]，不使用真正 0 速 bin=16。
AZIMUTH_ZERO_DOPPLER_BIN = NUM_CHIRPS_PER_FRAME // 2
AZIMUTH_USE_NEAR_ZERO_DOPPLER = True
AZIMUTH_NEAR_ZERO_HALF_WIDTH = 2
AZIMUTH_EXCLUDE_ZERO_DOPPLER = True
AZIMUTH_NEAR_ZERO_WEIGHTED_SUM = True
AZIMUTH_ANGLE_FFT_BINS = SAVE_DOPPLER_BINS
AZIMUTH_USE_RELATIVE_DB_SCALE = True
AZIMUTH_DYNAMIC_RANGE_DB = 35.0
AZIMUTH_FALLBACK_VMIN_DB = 55.0
AZIMUTH_FALLBACK_VMAX_DB = 110.0

# 如果測試後發現「手往右，azimuth 亮塊往左」，把這個改成 True。
AZIMUTH_FLIP_LEFT_RIGHT = False

# ============================================================
# Azimuth processing settings: reference-code style
# ============================================================
# 這版改成參考 01_record_radar_camera_only.py 的流程：
#   zero-Doppler complex RX snapshot
#       -> optional complex background subtraction
#       -> RX pair angle FFT
#       -> angle-axis foreground enhancement
#       -> relative dB fixed floor display
#
# 注意：這裡不再做上一版很強的 range gate / row median / hard threshold，
# 避免把目標壓到幾乎看不見。
AZIMUTH_DENOISE_ENABLE = True

# complex background subtraction, similar to reference code's BG_ENABLED / update_complex_background.
# 訓練資料建議先關掉，避免慢速手部訊號被背景模型吃掉。
# 如果畫面太髒再改回 True，並建議 AZIMUTH_BG_ALPHA 使用 0.03~0.08。
AZIMUTH_COMPLEX_BG_ENABLE = False
AZIMUTH_BG_ALPHA = 0.05
AZIMUTH_BG_WARMUP_FRAMES = 60

# angle foreground enhancement, similar to reference code's enhance_angle_foreground().
AZIMUTH_FOREGROUND_ENABLE = True
AZIMUTH_FOREGROUND_MODE = "bright"       # "abs" / "bright" / "dark"
AZIMUTH_BASELINE_KERNEL = 11            # 對 angle 軸做 moving-average baseline；32 bins 建議 7~11

# relative dB display, similar to reference code's to_relative_db() + fixed floor.
AZIMUTH_DB_DISPLAY = True
AZIMUTH_DB_FLOOR = -15.0
AZIMUTH_DISPLAY_MAX = 64.0             # 顯示值會轉成 0~16，越亮代表越接近本幀峰值
AZIMUTH_MIN_ENERGY = 1.0

# 可選：只沿 angle 方向稍微變寬，避免格子太硬。0/1 = 關閉。
AZIMUTH_SMOOTH_ANGLE_KERNEL = 3
AZIMUTH_SMOOTH_RANGE_KERNEL = 1
AZIMUTH_SMOOTH_GAIN = 1.0

# 顯示端 temporal smoothing，可降低右圖閃爍。
# alpha 越大越黏前一幀；0 = 不用前一幀。
AZIMUTH_TEMPORAL_EMA_ALPHA = 0.0

# Camera 顯示方向。現在預設不左右翻轉，避免畫面跟實際左右相反。
CAMERA_DISPLAY_FLIP_HORIZONTAL = False

# 顏色使用 OpenCV JET colormap，接近你截圖中的藍底、綠黃紅高亮效果。
# 如果你之後想試更鮮豔，可以改成 cv2.COLORMAP_TURBO。
RDI_COLORMAP = cv2.COLORMAP_JET

# DCA1000 complex sample to ADC cube ordering.
# "ti_rx_block" 對齊 TI MATLAB parser 常見整理方式：
#   chirp -> RX -> ADC samples
# "sample_pair_rx" 用於測試另一種可能排列：
#   chirp -> sample_pair -> RX -> 2 samples
# 正式使用前建議用單一近距離目標測試哪一種 range peak 會隨距離穩定移動。
DCA1000_CUBE_ORDER = "ti_rx_block"


# ============================================================
# AIC background suppression settings
# ============================================================
# AIC_ON=True 時，會在 Range FFT / Doppler FFT 之前，
# 先對每個 RX 的 raw ADC IQ 做自適應背景扣除。
#
# AIC 形式對齊 map_visualize_meansub_window_AIC_demoboard_213_mouse.py：
#   AIC_out = raw - reference_bank
#   error   = mean(AIC_out over chirps)
#   reference_bank = reference_bank + (1 / W) * error
#
# W 越小，背景更新越快，但也越可能把慢速手部訊號學進背景；
# W 越大，背景更新越慢，但保留手部動態比較穩。
AIC_ON = True
AIC_W_START = 4
AIC_W_END = 4


# ============================================================
# DCA1000 CLI control
# 只用來做 fpga / record 設定，不用 start_record / stop_record
# ============================================================

def run_dca_command(command, check=False):
    """
    command:
        fpga
        record
        fpga_version
        query_status
        query_sys_status

    注意：
        這裡不要拿來跑 start_record / stop_record。
        start/stop record 會改用 Python 直接送 UDP command。
    """
    cmd = [DCA_CONTROL_EXE, command, DCA_JSON]

    print(f"[DCA CLI >>] {' '.join(cmd)}")
    print(f"[DCA CLI cwd] {DCA_DIR}")

    env = os.environ.copy()
    env["PATH"] = DCA_DIR + os.pathsep + env.get("PATH", "")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=False,
        cwd=DCA_DIR,
        env=env,
    )

    if result.stdout:
        print("[DCA CLI <<]")
        print(result.stdout.strip())

    if result.stderr:
        print("[DCA CLI ERR]")
        print(result.stderr.strip())

    if result.returncode != 0:
        print(f"[WARN] DCA CLI command '{command}' return code = {result.returncode}")
        if check:
            raise RuntimeError(f"DCA CLI command failed: {command}")

    return result.returncode


# ============================================================
# DCA1000 direct UDP start / stop
# ============================================================

def dca_send_udp_command(cmd_code, data=b"", timeout=2.0):
    """
    直接送 DCA1000 command packet 到 192.168.33.180:4096。

    Packet format:
        uint16 header = 0xA55A
        uint16 command code
        uint16 data size
        payload
        uint16 footer = 0xEEAA
    """
    packet = (
        struct.pack("<HHH", DCA_HEADER, cmd_code, len(data))
        + data
        + struct.pack("<H", DCA_FOOTER)
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)

    # 綁定到 PC command port，避免 Windows 從隨機 port 回應造成 DCA 沒法對應
    sock.bind((PC_IP, DCA_CMD_PORT))

    try:
        print(f"[DCA UDP >>] cmd=0x{cmd_code:02X}, packet_bytes={len(packet)}")
        sock.sendto(packet, (DCA_IP, DCA_CMD_PORT))

        resp, addr = sock.recvfrom(1024)
        print(f"[DCA UDP <<] from={addr}, bytes={len(resp)}, hex={resp.hex()}")

        if len(resp) < 8:
            print("[WARN] DCA response too short")
            return False

        # 常見回應格式：
        # header, cmd_code, status, footer
        header, resp_cmd, status, footer = struct.unpack("<HHHH", resp[:8])

        if header != DCA_HEADER or footer != DCA_FOOTER:
            print(
                f"[WARN] DCA response header/footer mismatch: "
                f"header=0x{header:04X}, footer=0x{footer:04X}"
            )

        if status == 0:
            print(f"[DCA UDP] cmd=0x{resp_cmd:02X} success")
            return True

        print(f"[WARN] DCA UDP command failed, resp_cmd=0x{resp_cmd:02X}, status={status}")
        return False

    finally:
        sock.close()


def dca_start_record_direct():
    ok = dca_send_udp_command(CMD_RECORD_START)
    if not ok:
        raise RuntimeError("DCA direct RECORD_START failed")


def dca_stop_record_direct():
    try:
        dca_send_udp_command(CMD_RECORD_STOP)
    except Exception as e:
        print(f"[WARN] DCA direct RECORD_STOP failed: {e}")


# ============================================================
# Radar CLI helpers
# ============================================================

def read_until_response(ser, timeout=CMD_TIMEOUT_SEC):
    """
    讀取 CLI 回應直到 Done / Error / mmwDemo:/> / timeout。
    """
    t0 = time.time()
    lines = []
    buffer = ""

    while time.time() - t0 < timeout:
        n = ser.in_waiting

        if n > 0:
            data = ser.read(n).decode("utf-8", errors="ignore")
            buffer += data

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip("\r").strip()

                if line:
                    print(f"[RADAR <<] {line}")
                    lines.append(line)

                    if "Error" in line:
                        return lines, "error"

                    if line == "Done":
                        return lines, "done"

                    if "mmwDemo:/>" in line:
                        return lines, "prompt"

        time.sleep(0.02)

    leftover = buffer.strip()
    if leftover:
        print(f"[RADAR <<] {leftover}")
        lines.append(leftover)

        if "Error" in leftover:
            return lines, "error"
        if "Done" in leftover:
            return lines, "done"
        if "mmwDemo:/>" in leftover:
            return lines, "prompt"

    return lines, "timeout"


def send_radar_line(ser, line, timeout=CMD_TIMEOUT_SEC):
    line = line.strip()
    if not line:
        return "skip"

    print(f"[RADAR >>] {line}")
    ser.write((line + "\n").encode("utf-8"))

    lines, status = read_until_response(ser, timeout=timeout)

    if status == "timeout":
        print(f"[WARN] Timeout waiting response for: {line}")

    if status == "error":
        print(f"[ERROR] Radar command failed: {line}")

    return status


def load_cfg_lines(cfg_path):
    cfg_path = Path(cfg_path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find cfg file: {cfg_path}")

    lines = []

    for raw in cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()

        if not line:
            continue

        if line.startswith("%") or line.startswith("#"):
            continue

        lines.append(line)

    return lines


def open_radar_cli():
    ser = serial.Serial(CLI_PORT, CLI_BAUD, timeout=0.2)
    time.sleep(0.5)

    print(f"[RADAR] Open {CLI_PORT} @ {CLI_BAUD}")

    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(0.2)

    print("[RADAR >>] <ENTER>")
    ser.write(b"\n")
    read_until_response(ser, timeout=1.0)

    return ser


def send_radar_cfg_without_sensor_start(ser):
    """
    送 cfg，但跳過 sensorStart。
    因為流程要先啟動 DCA1000 record，再讓 radar sensorStart。
    """
    cfg_lines = load_cfg_lines(RADAR_CFG)

    print("====================================")
    print("Send radar cfg without sensorStart")
    print("====================================")
    print(f"CFG      : {RADAR_CFG}")
    print(f"Commands : {len(cfg_lines)}")
    print("====================================")

    error_count = 0
    timeout_count = 0

    for line in cfg_lines:
        if line.strip() == "sensorStart":
            print("[RADAR] Skip sensorStart for now.")
            continue

        status = send_radar_line(ser, line)

        if status == "error":
            error_count += 1

            if not line.startswith("sensorStop"):
                raise RuntimeError(f"Radar cfg command failed: {line}")

        elif status == "timeout":
            timeout_count += 1

    print("====================================")
    print("Radar cfg sent")
    print(f"errors   : {error_count}")
    print(f"timeouts : {timeout_count}")
    print("====================================")


def radar_sensor_start(ser):
    print("====================================")
    print("Radar sensorStart")
    print("====================================")
    status = send_radar_line(ser, "sensorStart", timeout=5.0)

    if status == "error":
        raise RuntimeError("sensorStart failed")

    print("[RADAR] sensorStart sent.")


def radar_sensor_stop(ser):
    try:
        print("====================================")
        print("Radar sensorStop")
        print("====================================")
        send_radar_line(ser, "sensorStop", timeout=2.0)
    except Exception as e:
        print(f"[WARN] radar sensorStop failed: {e}")


# ============================================================
# Raw ADC parsing / AIC / RDI processing
# ============================================================

class AICBackgroundRemover:
    """
    對每個 RX 的 raw ADC IQ 做自適應干擾/背景扣除。

    輸入 adc_cube.shape = (chirps, rx, samples)
    內部會把每個 RX 轉成 raw.shape = (samples, chirps)，
    對齊舊版 map_gen_module.run() 裡 AIC 的資料方向：
        raw = rawdata[:upsample_num, :, ch]

    AIC 更新：
        AIC_out = raw - reference_bank
        error   = mean(AIC_out, axis=chirp)
        reference_bank = reference_bank + alpha * error
        alpha = 1 / W

    回傳：
        aic_cube.shape = (chirps, rx, samples)
    """

    def __init__(self, num_adc_samples, num_rx, w_start=4, w_end=4, enabled=True):
        self.num_adc_samples = int(num_adc_samples)
        self.num_rx = int(num_rx)
        self.w_start = int(w_start)
        self.w_end = int(w_end)
        self.enabled = bool(enabled)

        if self.w_start <= 0 or self.w_end <= 0:
            raise ValueError("AIC_W_START and AIC_W_END must be positive integers")

        self.W = self.w_start
        self.reference_bank = np.zeros(
            (self.num_adc_samples, self.num_rx),
            dtype=np.complex64,
        )
        self.frame_count = 0

    def reset(self):
        self.W = self.w_start
        self.reference_bank.fill(0)
        self.frame_count = 0

    def apply(self, adc_cube):
        """
        將 AIC 套用在 raw ADC IQ 上。

        adc_cube.shape 必須是：
            (NUM_CHIRPS_PER_FRAME, NUM_RX, NUM_ADC_SAMPLES)
        """
        if not self.enabled:
            return adc_cube

        if adc_cube.shape != (NUM_CHIRPS_PER_FRAME, NUM_RX, NUM_ADC_SAMPLES):
            raise ValueError(
                f"adc_cube shape mismatch: got {adc_cube.shape}, "
                f"expected {(NUM_CHIRPS_PER_FRAME, NUM_RX, NUM_ADC_SAMPLES)}"
            )

        aic_cube = np.empty_like(adc_cube, dtype=np.complex64)
        alpha = np.float32(1.0 / float(self.W))

        for rx in range(self.num_rx):
            # adc_cube[:, rx, :] 是 (chirps, samples)
            # 轉成 (samples, chirps)，與舊版 AIC 的 raw 方向一致。
            raw = adc_cube[:, rx, :].T.astype(np.complex64, copy=False)

            # reference_bank[:, rx:rx+1] 是 (samples, 1)，
            # 透過 broadcasting 對所有 chirps 扣同一條背景參考。
            ref_col = self.reference_bank[:, rx:rx + 1]
            aic_out = raw - ref_col

            # 沿 chirp 維度取平均，得到每個 fast-time sample 的背景誤差。
            error_bank = np.mean(aic_out, axis=1, keepdims=True).astype(np.complex64)

            # 更新 reference bank。
            self.reference_bank[:, rx:rx + 1] = ref_col + alpha * error_bank

            # 轉回 (chirps, samples)，讓後面的 FFT 流程不用改。
            aic_cube[:, rx, :] = aic_out.T

        if self.W < self.w_end:
            self.W += 1

        self.frame_count += 1
        return aic_cube


def parse_one_frame_iwr6843_dca1000_complex(frame_bytes):
    """
    Parse IWR6843 / xWR16xx + DCA1000 complex ADC data.

    TI SWRA581B parser logic for xWR16xx/IWR6843 + DCA1000:
        - DCA1000 data is int16 two's-complement.
        - Complex stream in file is: I1, I2, Q1, Q2, I3, I4, Q3, Q4, ...
        - After I/Q combination, reshape one chirp as:
              RX0 sample 0..N-1,
              RX1 sample 0..N-1,
              RX2 sample 0..N-1,
              RX3 sample 0..N-1.

    Return:
        adc_cube.shape = (chirps, rx, samples)
    """
    raw_i16 = np.frombuffer(frame_bytes, dtype=np.int16)

    expected_i16 = NUM_CHIRPS_PER_FRAME * NUM_RX * NUM_ADC_SAMPLES * 2

    if raw_i16.size != expected_i16:
        raise ValueError(
            f"Frame int16 count mismatch: got {raw_i16.size}, expected {expected_i16}"
        )

    if raw_i16.size % 4 != 0:
        raise ValueError(f"Raw int16 size must be multiple of 4, got {raw_i16.size}")

    if NUM_ADC_SAMPLES % 2 != 0:
        raise ValueError("NUM_ADC_SAMPLES must be even for DCA1000 I1,I2,Q1,Q2 grouping")

    # Every 4 int16 values are [I1, I2, Q1, Q2].
    raw4 = raw_i16.reshape(-1, 4)

    c0 = raw4[:, 0].astype(np.float32) + 1j * raw4[:, 2].astype(np.float32)
    c1 = raw4[:, 1].astype(np.float32) + 1j * raw4[:, 3].astype(np.float32)

    complex_samples = np.empty(raw_i16.size // 2, dtype=np.complex64)
    complex_samples[0::2] = c0
    complex_samples[1::2] = c1

    # TI MATLAB example does:
    #   LVDS = reshape(LVDS, numADCSamples*numRX, numChirps).';
    # then each RX row is a contiguous block of numADCSamples inside each chirp.
    adc_cube = complex_samples.reshape(
        NUM_CHIRPS_PER_FRAME,
        NUM_RX,
        NUM_ADC_SAMPLES,
    )

    return adc_cube.astype(np.complex64)

def compute_rdi_per_rx(adc_cube):
    """
    adc_cube.shape = (chirps, rx, samples)

    若 AIC_ON=True，傳進來的 adc_cube 已經是 AIC 扣背景後的 raw ADC IQ。

    重要：這裡不再做 NUM_ADC_SAMPLES // 2 裁切。
    IWR6843 的 ADC 是 complex I/Q chain；complex FFT 不具有 real-only FFT 的共軛對稱。
    這版會在 Range FFT 前做 2x zero-padding：32 samples -> 64-point Range FFT。

    回傳：
        rdi_complex.shape = (rx, range_bins, doppler_bins)
        = (4, 64, 32)
    """
    if adc_cube.shape != (NUM_CHIRPS_PER_FRAME, NUM_RX, NUM_ADC_SAMPLES):
        raise ValueError(
            f"adc_cube shape mismatch: got {adc_cube.shape}, "
            f"expected {(NUM_CHIRPS_PER_FRAME, NUM_RX, NUM_ADC_SAMPLES)}"
        )

    # 1) Range FFT along fast-time / ADC sample axis.
    #    2x zero-padding: 32 ADC samples -> 64-point Range FFT.
    #    Full padded FFT length is kept; NO half-spectrum cropping here.
    range_fft = np.fft.fft(adc_cube, n=RANGE_FFT_SIZE, axis=2)

    # 2) Optional static clutter removal along slow-time / chirp axis.
    #    This suppresses the zero-Doppler vertical bright line.
    if STATIC_CLUTTER_REMOVAL:
        range_fft = range_fft - np.mean(range_fft, axis=0, keepdims=True)

    # 3) Doppler FFT along slow-time / chirp axis.
    doppler_fft = np.fft.fft(range_fft, n=NUM_CHIRPS_PER_FRAME, axis=0)
    doppler_fft = np.fft.fftshift(doppler_fft, axes=0)

    # (doppler, rx, range) -> (rx, range, doppler)
    rdi_complex = np.transpose(doppler_fft, (1, 2, 0))

    return rdi_complex.astype(np.complex64)


def crop_rdi_for_model(rdi_complex):
    """
    目前 cfg 是 4 RX，所以輸出：
        real/imag split = 8 channels
        model_input.shape = (8, 64, 32)
    """
    rdi_crop = rdi_complex[
        :,
        :SAVE_RANGE_BINS,
        :SAVE_DOPPLER_BINS,
    ]  # (4, 64, 32)

    model_input = np.concatenate(
        [rdi_crop.real, rdi_crop.imag],
        axis=0,
    )

    return model_input.astype(np.float32)


def rdi_complex_to_db(rdi_complex):
    """
    將 per-RX complex RDI 轉成 dB magnitude，方便即時顯示。

    input:
        rdi_complex.shape = (rx, range_bins, doppler_bins)
    output:
        rdi_db.shape      = (rx, range_bins, doppler_bins)
    """
    mag = np.abs(rdi_complex).astype(np.float32)
    rdi_db = 20.0 * np.log10(mag + RDI_DISPLAY_EPS)
    return rdi_db.astype(np.float32)



class RuntimeRDIControls:
    """
    保留給 receiver 使用的狀態物件，但不再提供畫面上的滑桿或勾選框。

    目前 AIC 是否啟用完全由程式最上方的 AIC_ON 預設值決定。
    Complex background subtraction display 已移除；dB 顯示範圍也固定使用：
        RDI_FIXED_VMIN_DB / RDI_FIXED_VMAX_DB
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.use_aic = bool(AIC_ON)

    def snapshot(self):
        with self.lock:
            return {"use_aic": bool(self.use_aic)}


_AZIMUTH_BG_SNAPSHOT = None
_AZIMUTH_BG_FRAME_COUNT = 0


def reset_azimuth_background():
    """Reset azimuth complex background model used for display/recording."""
    global _AZIMUTH_BG_SNAPSHOT, _AZIMUTH_BG_FRAME_COUNT
    _AZIMUTH_BG_SNAPSHOT = None
    _AZIMUTH_BG_FRAME_COUNT = 0


def update_azimuth_complex_background(current_snapshot):
    """
    Reference-code style complex background subtraction.

    current_snapshot.shape = (range_bins, 2)
    Returns residual snapshot. During warmup, returns zeros so background can settle.
    """
    global _AZIMUTH_BG_SNAPSHOT, _AZIMUTH_BG_FRAME_COUNT

    cur = np.asarray(current_snapshot, dtype=np.complex64)

    if not AZIMUTH_COMPLEX_BG_ENABLE:
        return cur

    if _AZIMUTH_BG_SNAPSHOT is None or _AZIMUTH_BG_SNAPSHOT.shape != cur.shape:
        _AZIMUTH_BG_SNAPSHOT = cur.copy()
        _AZIMUTH_BG_FRAME_COUNT = 1
        return np.zeros_like(cur, dtype=np.complex64)

    residual = cur - _AZIMUTH_BG_SNAPSHOT
    alpha = float(AZIMUTH_BG_ALPHA)
    alpha = max(0.0, min(1.0, alpha))
    _AZIMUTH_BG_SNAPSHOT = ((1.0 - alpha) * _AZIMUTH_BG_SNAPSHOT + alpha * cur).astype(np.complex64)
    _AZIMUTH_BG_FRAME_COUNT += 1

    if _AZIMUTH_BG_FRAME_COUNT <= int(AZIMUTH_BG_WARMUP_FRAMES):
        return np.zeros_like(cur, dtype=np.complex64)

    return residual.astype(np.complex64)


def smooth_rows_for_azimuth(mat, kernel_size):
    """Smooth each range row along the angle axis, matching the reference program idea."""
    mat = np.asarray(mat, dtype=np.float32)
    k = int(kernel_size)
    if k <= 1:
        return mat.copy()
    if k % 2 == 0:
        k += 1

    pad = k // 2
    kernel = np.ones(k, dtype=np.float32) / float(k)
    out = np.empty_like(mat, dtype=np.float32)
    for r in range(mat.shape[0]):
        row_pad = np.pad(mat[r], (pad, pad), mode="edge")
        out[r] = np.convolve(row_pad, kernel, mode="valid")
    return out.astype(np.float32)


def enhance_azimuth_foreground(angle_map):
    """
    Foreground enhancement copied conceptually from the reference script:
        baseline = smooth_rows(angle_map)
        abs / bright / dark residual
    """
    x = np.asarray(angle_map, dtype=np.float32)
    if not AZIMUTH_FOREGROUND_ENABLE:
        return x

    baseline = smooth_rows_for_azimuth(x, AZIMUTH_BASELINE_KERNEL)
    mode = str(AZIMUTH_FOREGROUND_MODE).lower()
    if mode == "bright":
        fg = x - baseline
    elif mode == "dark":
        fg = baseline - x
    else:
        fg = np.abs(x - baseline)

    fg = np.clip(fg, 0.0, None)
    return fg.astype(np.float32)


def smooth_and_widen_azimuth(angle_map):
    """Small display smoothing. It is intentionally gentle so the target is not erased."""
    out = np.asarray(angle_map, dtype=np.float32)

    k_angle = int(AZIMUTH_SMOOTH_ANGLE_KERNEL)
    if k_angle > 1:
        out = smooth_rows_for_azimuth(out, k_angle)

    k_range = int(AZIMUTH_SMOOTH_RANGE_KERNEL)
    if k_range > 1:
        if k_range % 2 == 0:
            k_range += 1
        out = cv2.GaussianBlur(out.astype(np.float32), (1, k_range), 0)

    out = out * float(AZIMUTH_SMOOTH_GAIN)
    return out.astype(np.float32)


def azimuth_to_reference_display_scale(angle_map):
    """
    Convert angle magnitude to a stable display scale 0~AZIMUTH_DISPLAY_MAX.
    This follows the reference program's relative dB display idea:
        20log10(x) -> subtract max -> clip to floor -> map to 0~1.
    """
    x = np.asarray(angle_map, dtype=np.float32)
    if x.size == 0:
        return x

    if (not np.any(np.isfinite(x))) or float(np.nanmax(x)) < float(AZIMUTH_MIN_ENERGY):
        return np.zeros_like(x, dtype=np.float32)

    if AZIMUTH_DB_DISPLAY:
        x_db = 20.0 * np.log10(np.maximum(x, 1e-6))
        max_db = float(np.nanmax(x_db))
        if not np.isfinite(max_db):
            return np.zeros_like(x, dtype=np.float32)
        x_db = x_db - max_db
        floor_db = float(AZIMUTH_DB_FLOOR)
        x_db = np.clip(x_db, floor_db, 0.0)
        x_norm = (x_db - floor_db) / max(0.0 - floor_db, 1e-6)
        return (x_norm * float(AZIMUTH_DISPLAY_MAX)).astype(np.float32)

    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    vmax = float(np.percentile(x[finite], 99.5))
    vmin = float(np.percentile(x[finite], 1.0))
    if vmax <= vmin:
        return np.zeros_like(x, dtype=np.float32)
    x_norm = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
    return (x_norm * float(AZIMUTH_DISPLAY_MAX)).astype(np.float32)


def get_near_zero_doppler_bins(doppler_bins):
    """
    取得 near-zero Doppler bins，但可排除真正 zero-Doppler bin。

    doppler_bins=32 時：
        center = 16
        half_width = 2
        exclude center -> [14, 15, 17, 18]
    """
    doppler_bins = int(doppler_bins)
    center = doppler_bins // 2
    half = int(AZIMUTH_NEAR_ZERO_HALF_WIDTH)

    start = max(0, center - half)
    end = min(doppler_bins, center + half + 1)

    bins = list(range(start, end))

    if AZIMUTH_EXCLUDE_ZERO_DOPPLER:
        bins = [b for b in bins if b != center]

    # 保險：如果 half_width 太小導致空集合，就退回 center±1。
    if len(bins) == 0:
        bins = []
        if center - 1 >= 0:
            bins.append(center - 1)
        if center + 1 < doppler_bins:
            bins.append(center + 1)

    # 再保險：極端情況下至少回傳 center。
    if len(bins) == 0:
        bins = [center]

    return np.array(bins, dtype=np.int32)


def compute_azimuth_map_from_two_rdi(rdi_complex):
    """
    用兩個 RX 的 complex RDI 產生 near-zero-Doppler azimuth heatmap。

    新流程：
        1. 不取真正 zero-Doppler bin。
        2. 取中心附近的 near-zero Doppler bins，例如 [14, 15, 17, 18]。
        3. 依照能量做 weighted complex sum，保留 RX 間 phase。
        4. 對 RX pair 做 angle FFT。
        5. 輸出 shape = (range_bins, doppler_bins)，方便和 RDI 疊成 2-channel input。

    注意：
        - 若 AZIMUTH_USE_NEAR_ZERO_DOPPLER=False，才會退回舊版 zero-Doppler 作法。
        - 如果手往右但亮塊往左，請把 AZIMUTH_FLIP_LEFT_RIGHT 改成 True。
    """
    if rdi_complex is None or rdi_complex.shape[0] < 2:
        raise ValueError("Need at least two RX complex RDI maps to compute azimuth map.")

    range_bins = int(rdi_complex.shape[1])
    doppler_bins = int(rdi_complex.shape[2])

    if AZIMUTH_USE_NEAR_ZERO_DOPPLER:
        near_bins = get_near_zero_doppler_bins(doppler_bins)

        # 取 RX0/RX1 在 near-zero Doppler bins 的 complex data。
        # shape: (2, range_bins, num_near_bins)
        near_cube = rdi_complex[:2, :, near_bins].astype(np.complex64, copy=False)

        if AZIMUTH_NEAR_ZERO_WEIGHTED_SUM:
            # 用每個 range、每個 Doppler bin 的平均能量當權重。
            # power shape: (range_bins, num_near_bins)
            power = np.mean(np.abs(near_cube) ** 2, axis=0).astype(np.float32)

            # 避免權重全部為 0。
            weights = power / (np.sum(power, axis=1, keepdims=True) + 1e-6)

            # 對 near-zero Doppler bins 做 complex weighted sum。
            # 這裡不能先取 magnitude 再平均，否則 RX 間相位差會被破壞。
            # output before transpose: (2, range_bins)
            ant_snapshot = np.sum(
                near_cube * weights[None, :, :],
                axis=2,
            ).T.astype(np.complex64)  # shape: (range_bins, 2)

        else:
            # 簡單 complex mean。仍然保留相位，比 magnitude mean 更適合做 angle FFT。
            ant_snapshot = np.mean(near_cube, axis=2).T.astype(np.complex64)

    else:
        # 舊版：固定取真正 zero-Doppler bin。不建議拿來訓練角度。
        zero_bin = int(AZIMUTH_ZERO_DOPPLER_BIN)
        zero_bin = max(0, min(zero_bin, doppler_bins - 1))

        ant_snapshot = np.stack(
            [
                rdi_complex[0, :, zero_bin],
                rdi_complex[1, :, zero_bin],
            ],
            axis=1,
        ).astype(np.complex64, copy=False)

    # 訓練資料預設關閉 AZIMUTH_COMPLEX_BG_ENABLE，避免慢速手部訊號被扣掉。
    # 如果你只想改善即時顯示，可以再打開，但建議 alpha 不要太大。
    ant_snapshot = update_azimuth_complex_background(ant_snapshot)

    angle_bins = max(2, int(AZIMUTH_ANGLE_FFT_BINS))
    az_fft = np.fft.fftshift(
        np.fft.fft(ant_snapshot, n=angle_bins, axis=1),
        axes=1,
    )
    angle_mag = np.abs(az_fft).astype(np.float32)

    if AZIMUTH_FLIP_LEFT_RIGHT:
        angle_mag = np.fliplr(angle_mag)

    angle_mag = enhance_azimuth_foreground(angle_mag)
    angle_mag = smooth_and_widen_azimuth(angle_mag)
    az_show = azimuth_to_reference_display_scale(angle_mag)

    # 保險：如果 angle FFT bins 跟 RDI doppler bins 不同，resize 回 RDI 同寬。
    if az_show.shape[1] != doppler_bins:
        az_show = cv2.resize(
            az_show.astype(np.float32),
            (doppler_bins, range_bins),
            interpolation=cv2.INTER_LINEAR,
        )

    return az_show.astype(np.float32)

def make_two_channel_radar_frame(rdi_complex, display_range_start=None, display_range_end=None):
    """
    產生錄製用的 2-channel radar frame：
        channel 0: 兩個 RX magnitude 平均後的 RDI dB
        channel 1: near-zero-Doppler 兩 RX antenna FFT 得到的 azimuth heatmap

    output shape = (2, range_bins, doppler_bins)
    """
    mag_mean = np.mean(np.abs(rdi_complex[:2]).astype(np.float32), axis=0)
    rdi_db = 20.0 * np.log10(mag_mean + RDI_DISPLAY_EPS).astype(np.float32)
    azimuth_deg = compute_azimuth_map_from_two_rdi(rdi_complex)

    if display_range_start is not None or display_range_end is not None:
        start = 0 if display_range_start is None else int(display_range_start)
        end = rdi_db.shape[0] if display_range_end is None else int(display_range_end)
        rdi_db = rdi_db[start:end, :]
        azimuth_deg = azimuth_deg[start:end, :]

    return np.stack([rdi_db.astype(np.float32), azimuth_deg.astype(np.float32)], axis=0)


class RealtimeRDIViewer:
    """
    OpenCV 即時顯示兩個格子：
        左：兩個 RX 的 RDI magnitude 平均圖
        右：zero-Doppler 兩 RX antenna FFT 的 azimuth 圖

    已移除：
        - 下方 integrated controls
        - AIC runtime checkbox
        - Complex background subtraction for display
        - 所有 dB runtime sliders
    """

    def __init__(self, num_rx, range_bins, doppler_bins, controls=None):
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not available. Please install opencv-python or set SHOW_RDI_REALTIME = False."
            )

        if int(num_rx) < 2:
            raise ValueError("This viewer needs at least 2 RX RDI maps to compute the azimuth/PHD map.")

        self.num_rx = int(num_rx)
        self.controls = controls if controls is not None else RuntimeRDIControls()
        self.full_range_bins = int(range_bins)
        self.display_range_start, self.display_range_end = self._get_display_range_slice(self.full_range_bins)
        self.range_bins = int(self.display_range_end - self.display_range_start)
        self.doppler_bins = int(doppler_bins)

        self.last_frame_id = -1
        self.last_update_time = 0.0
        self.closed = False
        self.last_key = -1
        self.azimuth_ema = None
        self.last_base_canvas = None

        self.tile_w = int(RDI_TILE_WIDTH)
        self.tile_h = int(RDI_TILE_HEIGHT)
        self.gap = int(RDI_TILE_GAP)
        self.border = int(RDI_TILE_BORDER)
        self.title_bar_h = int(RDI_TITLE_BAR_HEIGHT)

        self.grid_w = self.tile_w * 2 + self.gap
        self.grid_h = self.tile_h

        # Camera 直接整合在同一個 OpenCV 視窗的下半部，不再另外開 camera 視窗。
        self.camera_w = int(CAMERA_DISPLAY_WIDTH)
        self.camera_h = int(CAMERA_DISPLAY_HEIGHT)
        self.camera_gap = 18
        self.canvas_w = max(self.border * 2 + self.grid_w, self.border * 2 + self.camera_w, 980)
        self.canvas_h = (
            self.title_bar_h
            + self.border
            + self.grid_h
            + self.camera_gap
            + self.camera_h
            + self.border
            + 30
        )
        self.grid_x0 = (self.canvas_w - self.grid_w) // 2
        self.grid_y0 = self.title_bar_h + self.border
        self.camera_x0 = (self.canvas_w - self.camera_w) // 2
        self.camera_y0 = self.grid_y0 + self.grid_h + self.camera_gap

        cv2.namedWindow(RDI_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RDI_WINDOW_NAME, self.canvas_w, self.canvas_h)

    def _get_display_range_slice(self, full_range_bins):
        full_range_bins = int(full_range_bins)
        mode = str(RDI_DISPLAY_RANGE_MODE).lower()
        half = full_range_bins // 2

        if mode == "full":
            start, end = 0, full_range_bins
        elif mode == "upper_half_screen":
            if RDI_DISPLAY_ORIGIN.lower() == "lower":
                start, end = half, full_range_bins
            else:
                start, end = 0, half
        elif mode == "lower_half_screen":
            if RDI_DISPLAY_ORIGIN.lower() == "lower":
                start, end = 0, half
            else:
                start, end = half, full_range_bins
        elif mode == "custom":
            start = int(RDI_DISPLAY_RANGE_START_BIN)
            end = full_range_bins if RDI_DISPLAY_RANGE_END_BIN is None else int(RDI_DISPLAY_RANGE_END_BIN)
        else:
            raise ValueError(
                f"Unknown RDI_DISPLAY_RANGE_MODE={RDI_DISPLAY_RANGE_MODE}. "
                "Use full / upper_half_screen / lower_half_screen / custom."
            )

        start = max(0, min(start, full_range_bins - 1))
        end = max(start + 1, min(end, full_range_bins))
        return start, end

    def _crop_display_range(self, img):
        return img[self.display_range_start:self.display_range_end, :]

    def _normalize_to_uint8(self, img, vmin, vmax):
        img = np.asarray(img, dtype=np.float32)
        img = np.nan_to_num(img, nan=vmin, posinf=vmax, neginf=vmin)
        img = np.clip(img, vmin, vmax)
        denom = max(float(vmax) - float(vmin), 1e-6)
        return ((img - float(vmin)) / denom * 255.0).astype(np.uint8)

    def _find_peak(self, img):
        work = np.asarray(img, dtype=np.float32).copy()
        work[~np.isfinite(work)] = -np.inf
        if not np.any(np.isfinite(work)):
            return 0, 0, 0.0
        idx = int(np.nanargmax(work))
        r_bin, d_bin = np.unravel_index(idx, work.shape)
        return int(r_bin), int(d_bin), float(img[r_bin, d_bin])

    def _make_heatmap_tile(self, img, title, subtitle, vmin, vmax, unit_text=""):
        img_show = np.asarray(img, dtype=np.float32).copy()
        peak_r, peak_d, peak_val = self._find_peak(img_show)
        peak_r_original = peak_r + self.display_range_start

        if RDI_DISPLAY_ORIGIN.lower() == "lower":
            img_show = np.flipud(img_show)
            peak_y_bin = self.range_bins - 1 - peak_r
        else:
            peak_y_bin = peak_r

        img_u8 = self._normalize_to_uint8(img_show, vmin, vmax)
        img_u8 = cv2.resize(img_u8, (self.tile_w, self.tile_h), interpolation=cv2.INTER_NEAREST)
        heatmap = cv2.applyColorMap(img_u8, RDI_COLORMAP)

        if RDI_DRAW_PEAK_MARKER:
            px = int((peak_d + 0.5) * self.tile_w / max(self.doppler_bins, 1))
            py = int((peak_y_bin + 0.5) * self.tile_h / max(self.range_bins, 1))
            cv2.drawMarker(
                heatmap,
                (px, py),
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        cv2.putText(heatmap, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(heatmap, subtitle, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(
            heatmap,
            f"peak r={peak_r_original}, d={peak_d}, {peak_val:.1f}{unit_text}",
            (12, self.tile_h - 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        near_text = "near=bottom" if RDI_DISPLAY_ORIGIN.lower() == "lower" else "near=top"
        cv2.putText(
            heatmap,
            f"Range x Doppler ({near_text}) | bins {self.display_range_start}-{self.display_range_end - 1}",
            (12, self.tile_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return heatmap

    def make_display_images(self, rdi_complex):
        radar_frame = make_two_channel_radar_frame(rdi_complex)
        rdi_db = self._crop_display_range(radar_frame[0])
        azimuth_db = self._crop_display_range(radar_frame[1])
        return rdi_db, azimuth_db, radar_frame

    def update(self, rdi_complex, frame_id, camera_panel=None, force=False):
        if self.closed:
            return None

        now = time.time()
        display_interval = 1.0 / max(float(CAMERA_DISPLAY_FPS), 1.0) if force else RDI_DISPLAY_INTERVAL_SEC
        if now - self.last_update_time < display_interval:
            return None
        same_radar_frame = frame_id == self.last_frame_id
        if same_radar_frame and not force:
            return None

        radar_frame = None
        if same_radar_frame and self.last_base_canvas is not None:
            canvas = self.last_base_canvas.copy()
        else:
            rdi_db, azimuth_db, radar_frame = self.make_display_images(rdi_complex)

        # Azimuth 已經依參考程式轉成 relative display scale：0~AZIMUTH_DISPLAY_MAX。
        # 這裡只做溫和 temporal EMA，降低閃爍，不再做 hard range gate。
        if AZIMUTH_DENOISE_ENABLE:
            alpha = float(AZIMUTH_TEMPORAL_EMA_ALPHA)
            alpha = max(0.0, min(1.0, alpha))
            if self.azimuth_ema is None or self.azimuth_ema.shape != azimuth_db.shape:
                self.azimuth_ema = azimuth_db.astype(np.float32).copy()
            else:
                self.azimuth_ema = alpha * self.azimuth_ema + (1.0 - alpha) * azimuth_db.astype(np.float32)
            azimuth_db = self.azimuth_ema.astype(np.float32)
            az_vmin = 0.0
            az_vmax = float(AZIMUTH_DISPLAY_MAX)
            az_unit = ""
        elif AZIMUTH_USE_RELATIVE_DB_SCALE and np.any(np.isfinite(azimuth_db)):
            az_vmax = float(np.nanpercentile(azimuth_db, 99.0))
            az_vmin = az_vmax - float(AZIMUTH_DYNAMIC_RANGE_DB)
            az_unit = " dB"
        else:
            az_vmin = float(AZIMUTH_FALLBACK_VMIN_DB)
            az_vmax = float(AZIMUTH_FALLBACK_VMAX_DB)
            az_unit = " dB"
        if az_vmax <= az_vmin:
            az_vmax = az_vmin + 1.0

        rdi_tile = self._make_heatmap_tile(
            rdi_db,
            "RDI",
            "mean magnitude of RX1/RX2",
            RDI_FIXED_VMIN_DB,
            RDI_FIXED_VMAX_DB,
            " dB",
        )
        az_tile = self._make_heatmap_tile(
            azimuth_db,
            "Azimuth",
            "near-zero Doppler angle FFT",
            az_vmin,
            az_vmax,
            az_unit,
        )

        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        ctrl = self.controls.snapshot()
        title = (
            f"Realtime RDI + Azimuth/PHD | frame={frame_id} | "
            f"AIC={'ON' if ctrl['use_aic'] else 'OFF'} | "
            f"fixed RDI scale={RDI_FIXED_VMIN_DB:.0f}~{RDI_FIXED_VMAX_DB:.0f} dB | "
            f"az near-zero={'ON' if AZIMUTH_USE_NEAR_ZERO_DOPPLER else 'OFF'} | "
            f"az denoise={'ON' if AZIMUTH_DENOISE_ENABLE else 'OFF'} | q/ESC close | r=input/start record | s=stop record"
        )
        cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        positions = [
            (self.grid_x0, self.grid_y0),
            (self.grid_x0 + self.tile_w + self.gap, self.grid_y0),
        ]
        for tile, (x, y) in zip([rdi_tile, az_tile], positions):
            canvas[y:y + self.tile_h, x:x + self.tile_w] = tile
            cv2.rectangle(canvas, (x - 1, y - 1), (x + self.tile_w, y + self.tile_h), (80, 80, 80), 1, cv2.LINE_AA)

        # Camera panel：放在 RDI / Azimuth 底下，同一個視窗顯示。
        if camera_panel is None:
            cam_panel = np.zeros((self.camera_h, self.camera_w, 3), dtype=np.uint8)
            cv2.putText(
                cam_panel,
                "Camera not ready / disabled",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
        else:
            cam_panel = cv2.resize(camera_panel, (self.camera_w, self.camera_h), interpolation=cv2.INTER_AREA)

        canvas[self.camera_y0:self.camera_y0 + self.camera_h, self.camera_x0:self.camera_x0 + self.camera_w] = cam_panel
        cv2.rectangle(
            canvas,
            (self.camera_x0 - 1, self.camera_y0 - 1),
            (self.camera_x0 + self.camera_w, self.camera_y0 + self.camera_h),
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "Display controls removed: values are taken from code defaults only. Camera is integrated below radar panels.",
            (10, self.canvas_h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(RDI_WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        self.last_key = key
        if key in (27, ord('q')):
            self.closed = True
            cv2.destroyWindow(RDI_WINDOW_NAME)

        self.last_frame_id = frame_id
        self.last_update_time = now
        display_record_frame = np.stack(
            [
                rdi_db.astype(np.float32),
                azimuth_db.astype(np.float32),
            ],
            axis=0,
        )

        return display_record_frame

    def close(self):
        self.closed = True
        if cv2 is not None:
            try:
                cv2.destroyWindow(RDI_WINDOW_NAME)
            except Exception:
                pass


# ============================================================
# AprilTag camera + radar-triggered recording
# ============================================================

ENABLE_APRILTAG_CAMERA = True
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_TARGET_FPS = 60.0
CAMERA_DISPLAY_FPS = 60.0
CAMERA_TAG_DETECTION_FPS = 20.0
CAMERA_DISPLAY_WIDTH = 960
CAMERA_DISPLAY_HEIGHT = 540
CALIB_PATH = "calib.npz"
APRILTAG_MARKER_LENGTH_M = 0.03
RECORD_FRAMES_DEFAULT = 5000
RECORD_VIDEO_FPS = 20.0
CAMERA_WINDOW_NAME = "AprilTag Camera Recording"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="DCA1000 realtime RDI/Azimuth viewer with AprilTag camera recording."
    )
    parser.add_argument(
        "--record-frames",
        type=int,
        default=RECORD_FRAMES_DEFAULT,
        help="Number of radar frames to save after pressing r.",
    )
    parser.add_argument(
        "--record-video-fps",
        type=float,
        default=RECORD_VIDEO_FPS,
        help="FPS used for the saved camera mp4.",
    )
    return parser


def ask_record_frame_count(default_frames):
    try:
        import tkinter as tk
        from tkinter import simpledialog
    except Exception as e:
        print(f"[REC] Cannot open frame-count input dialog: {e}")
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        root.update()
        value = simpledialog.askinteger(
            "Start Recording",
            "Radar frames to record:",
            parent=root,
            initialvalue=max(1, int(default_frames)),
            minvalue=1,
        )
    finally:
        root.destroy()

    if value is None:
        print("[REC] Recording cancelled.")
        return None
    return int(value)


class AprilTagCameraRecorder:
    """
    參考 dis_angle_recording.py 的 AprilTag 影像錄製流程：
        - 使用 AprilTag 0 / 1 / 3 計算角度與距離
        - radar frame 到達時同步儲存：H5 radar DS1 + camera mp4 + records.csv
        - 按 r 開始錄製 RECORD_FRAMES_DEFAULT 個 radar frames
        - 按 s 可提前結束並保存目前已錄到的資料
    """

    def __init__(self, enabled=True, record_video_fps=RECORD_VIDEO_FPS):
        self.enabled = bool(enabled)
        self.cap = None
        self.K = None
        self.dist = None
        self.cx = None
        self.markerLength = APRILTAG_MARKER_LENGTH_M
        self.align_px_thresh = 5
        self.baseline_v_thresh = 5
        self.record_video_fps = float(record_video_fps)

        self.detector = None
        self.latest_camera_frame = None
        self.latest_display_frame = None
        self.current_angle = np.nan
        self.current_distance = np.nan
        self.current_aligned = False

        self.is_recording = False
        self.total_frames = 0
        self.recorded_count = 0
        self.angles = []
        self.distances = []
        self.radar_frame_ids = []
        self.video_writer = None
        self.h5file = None
        self.h5ds = None
        self.record_dir = None

        if not self.enabled:
            return
        if cv2 is None:
            print("[CAM] OpenCV unavailable; AprilTag camera disabled.")
            self.enabled = False
            return
        if not Path(CALIB_PATH).exists():
            print(f"[CAM] Cannot find {CALIB_PATH}; AprilTag camera disabled.")
            self.enabled = False
            return

        calib = np.load(CALIB_PATH)
        self.K = calib["K"]
        self.dist = calib["dist"]
        self.cx = float(self.K[0, 2])

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if not self.cap.isOpened():
            print("[CAM] Failed to open camera; AprilTag camera disabled.")
            self.enabled = False
            return

        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dict, self.params)

        self.latest_display_frame = None
        print("[CAM] AprilTag camera recorder ready. Camera view is integrated in the main RDI window.")

    def update_camera(self):
        if not self.enabled or self.cap is None:
            return -1

        ret, frame = self.cap.read()
        if not ret:
            return -1

        self.latest_camera_frame = frame.copy()
        raw = frame.copy()
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)
        self.current_angle = np.nan
        self.current_distance = np.nan
        self.current_aligned = False

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
                    tvecs[int(tid)] = tvec_array[0, 0]
                    if int(tid) == 0:
                        rvec0 = rvecs[0, 0]

            if rvec0 is not None and {0, 1, 3}.issubset(tvecs):
                centers = {
                    int(tid): c.reshape(-1, 2).mean(axis=0)
                    for c, tid in zip(corners, ids.flatten()) if int(tid) in (0, 1)
                }
                u0, v0 = centers[0]
                u1, v1 = centers[1]
                if (abs(u0 - self.cx) < self.align_px_thresh and
                        abs(v1 - v0) < self.baseline_v_thresh):
                    self.current_aligned = True

                R_tag2cam, _ = cv2.Rodrigues(rvec0)
                tag_normal = R_tag2cam[:, 2]

                base = tvecs[1] - tvecs[0]
                base_proj = base - (base.dot(tag_normal)) * tag_normal
                base_norm = base_proj / max(np.linalg.norm(base_proj), 1e-9)

                forward = np.cross(tag_normal, base_norm)
                forward = forward / max(np.linalg.norm(forward), 1e-9)

                vec03 = tvecs[3] - tvecs[0]
                proj = vec03 - (vec03.dot(tag_normal)) * tag_normal

                dist_m = np.linalg.norm(proj)
                self.current_distance = round(float(dist_m * 100.0), 1)

                x = float(proj.dot(base_norm))
                y = float(proj.dot(forward))
                angle_rad = np.arctan2(x, y)
                self.current_angle = round(float(np.degrees(angle_rad)), 1)

        if CAMERA_DISPLAY_FLIP_HORIZONTAL:
            disp = cv2.flip(raw, 1)
        else:
            disp = raw.copy()
        dh, dw = disp.shape[:2]
        cv2.line(disp, (dw // 2, 0), (dw // 2, dh), (255, 255, 255), 1)

        angle_text = "-- deg" if np.isnan(self.current_angle) else f"{self.current_angle:.1f} deg"
        dist_text = "-- cm" if np.isnan(self.current_distance) else f"{self.current_distance:.1f} cm"
        rec_text = "REC" if self.is_recording else "READY"
        cv2.putText(disp, f"Angle: {angle_text} {'[Aligned]' if self.current_aligned else ''}",
                    (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f"Distance: {dist_text}",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f"{rec_text} {self.recorded_count}/{self.total_frames} | r=input/start | s=stop/save",
                    (20, dh - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

        self.latest_display_frame = disp.copy()
        return -1

    def get_display_frame(self):
        if self.latest_display_frame is None:
            return None
        return self.latest_display_frame.copy()

    def start_recording(self, total_frames, radar_frame_template):
        if not self.enabled:
            print("[CAM] Camera recorder is disabled.")
            return
        if self.is_recording:
            print("[REC] Already recording.")
            return
        if radar_frame_template is None:
            print("[REC] Radar frame not ready yet.")
            return
        if self.latest_camera_frame is None:
            print("[REC] Camera frame not ready yet.")
            return

        n = int(total_frames)
        if n <= 0:
            print("[REC] total_frames must be positive.")
            return

        self.angles = []
        self.distances = []
        self.radar_frame_ids = []
        self.total_frames = n
        self.recorded_count = 0
        self.is_recording = True

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.record_dir = os.path.join("Record", f"dca_angle_dist_record_{ts}")
        os.makedirs(self.record_dir, exist_ok=True)

        self.h5file = h5py.File(os.path.join(self.record_dir, f"data_{ts}.h5"), "w")
        c, h, w = radar_frame_template.shape
        self.h5ds = self.h5file.create_dataset("DS1", (n, c, h, w), maxshape=(None, c, h, w), dtype=np.float32)

        frame_h, frame_w = self.latest_camera_frame.shape[:2]
        self.video_writer = cv2.VideoWriter(
            os.path.join(self.record_dir, f"video_{ts}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.record_video_fps,
            (frame_w, frame_h),
        )
        if not self.video_writer.isOpened():
            print("[WARN] Camera video writer failed to open; mp4 will not be saved.")
            self.video_writer = None
        print(
            f"[REC] Started: {self.record_dir}, frames={n}, radar_shape={(c, h, w)}, "
            f"video_size=({frame_w}, {frame_h}), video_fps={self.record_video_fps:g}"
        )

    def record_radar_frame(self, frame_id, radar_frame):
        if not self.enabled or not self.is_recording:
            return
        if radar_frame is None or self.recorded_count >= self.total_frames:
            return

        self.h5ds[self.recorded_count, ...] = radar_frame.astype(np.float32)
        if self.latest_camera_frame is not None and self.video_writer is not None:
            self.video_writer.write(self.latest_camera_frame)

        self.radar_frame_ids.append(int(frame_id))
        self.angles.append(self.current_angle)
        self.distances.append(self.current_distance)
        self.recorded_count += 1

        if self.recorded_count % 50 == 0 or self.recorded_count == self.total_frames:
            print(f"[REC] {self.recorded_count}/{self.total_frames} radar frames saved")

        if self.recorded_count >= self.total_frames:
            self.finish_recording()

    def finish_recording(self):
        if not self.is_recording and self.h5file is None:
            return

        saved_n = int(self.recorded_count)
        self.is_recording = False

        if self.h5ds is not None and saved_n < self.total_frames:
            self.h5ds.resize((saved_n,) + self.h5ds.shape[1:])

        frames = np.arange(saved_n)
        ang_arr = np.array(self.angles[:saved_n], dtype=np.float32)
        dist_arr = np.array(self.distances[:saved_n], dtype=np.float32)
        for arr in (ang_arr, dist_arr):
            if arr.size == 0:
                continue
            mask = np.isnan(arr)
            if np.any(mask) and np.any(~mask):
                arr[mask] = np.interp(frames[mask], frames[~mask], arr[~mask])

        csv_path = None
        if self.record_dir is not None:
            csv_path = os.path.join(self.record_dir, "records.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["index", "radar_frame_id", "angle_deg", "distance_cm"])
                for i in range(saved_n):
                    a = ang_arr[i] if i < len(ang_arr) else np.nan
                    d = dist_arr[i] if i < len(dist_arr) else np.nan
                    fid = self.radar_frame_ids[i] if i < len(self.radar_frame_ids) else -1
                    w.writerow([i, fid, f"{a:.1f}", f"{d:.1f}"])

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.h5file is not None:
            self.h5file.close()
            self.h5file = None
            self.h5ds = None

        print(f"[REC] Done. Saved frames={saved_n}. CSV={csv_path}")

    def close(self):
        if self.is_recording or self.h5file is not None:
            self.finish_recording()
        if self.cap is not None:
            self.cap.release()
        # Camera view is integrated in the main RDI window, so no separate camera window is destroyed here.


# ============================================================
# UDP Receiver
# ============================================================

class DCA1000UDPReceiver:
    def __init__(self, controls=None):
        self.running = False
        self.thread = None

        self.total_packets = 0
        self.total_payload_bytes = 0
        self.frame_count = 0
        self.lost_packets = 0
        self.last_seq = None
        self.last_model_input = None
        self.controls = controls if controls is not None else RuntimeRDIControls()
        self._last_aic_enabled = self.controls.snapshot()["use_aic"]

        # 給 main thread 的即時 RDI 顯示用。
        # receiver thread 只更新最新一幀，不把所有 frame 堆在記憶體。
        self.latest_rdi_complex = None
        self.latest_rdi_frame_id = -1
        self.rdi_lock = threading.Lock()

        self.aic = AICBackgroundRemover(
            num_adc_samples=NUM_ADC_SAMPLES,
            num_rx=NUM_RX,
            w_start=AIC_W_START,
            w_end=AIC_W_END,
            enabled=self._last_aic_enabled,
        )

        self.raw_buffer = bytearray()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=5.0)

    def get_latest_rdi(self):
        """
        回傳目前最新一幀 complex RDI 的 copy，避免 GUI 顯示時剛好被接收執行緒改到。
        """
        with self.rdi_lock:
            if self.latest_rdi_complex is None:
                return None, -1

            return self.latest_rdi_complex.copy(), int(self.latest_rdi_frame_id)

    def _loop(self):
        print("====================================")
        print("DCA1000 UDP Receiver")
        print("====================================")
        print(f"Bind        : {PC_IP}:{DATA_PORT}")
        print(f"Frame bytes : {FRAME_BYTES:,}")
        print(f"ADC cube    : ({NUM_CHIRPS_PER_FRAME}, {NUM_RX}, {NUM_ADC_SAMPLES})")
        print(f"Range FFT   : n={RANGE_FFT_SIZE} (pad factor={RANGE_FFT_PAD_FACTOR})")
        print(f"AIC enabled : {self.controls.snapshot()['use_aic']}, W_START={AIC_W_START}, W_END={AIC_W_END}")
        print(f"Static clutter removal : {STATIC_CLUTTER_REMOVAL}")
        if AZIMUTH_USE_NEAR_ZERO_DOPPLER:
            print(f"Azimuth bins: near-zero {get_near_zero_doppler_bins(SAVE_DOPPLER_BINS).tolist()} "
                  f"(exclude zero={AZIMUTH_EXCLUDE_ZERO_DOPPLER})")
        else:
            print(f"Azimuth bin : zero-Doppler {AZIMUTH_ZERO_DOPPLER_BIN}")
        print(f"RDI         : ({NUM_RX}, {SAVE_RANGE_BINS}, {SAVE_DOPPLER_BINS})")
        print(f"Model input : ({NUM_RX * 2}, {SAVE_RANGE_BINS}, {SAVE_DOPPLER_BINS})")
        print("====================================")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        sock.bind((PC_IP, DATA_PORT))
        sock.settimeout(1.0)

        t0 = time.time()
        last_log = t0

        try:
            while self.running:
                try:
                    packet, addr = sock.recvfrom(9000)
                except socket.timeout:
                    continue

                if len(packet) <= DCA_PACKET_HEADER_BYTES:
                    continue

                self.total_packets += 1

                seq = struct.unpack("<I", packet[0:4])[0]

                if self.last_seq is not None:
                    expected = self.last_seq + 1

                    if seq != expected:
                        gap = seq - expected

                        if gap > 0:
                            self.lost_packets += gap
                            self.raw_buffer.clear()
                            self.last_seq = seq
                            print(
                                f"[WARN] Packet lost. expected={expected}, got={seq}, "
                                f"gap={gap}. Clear raw_buffer to avoid frame misalignment."
                            )
                            continue
                        else:
                            self.raw_buffer.clear()
                            self.last_seq = seq
                            print(
                                f"[WARN] Sequence backward/jump. expected={expected}, got={seq}. "
                                f"Clear raw_buffer."
                            )
                            continue

                self.last_seq = seq

                payload = packet[DCA_PACKET_HEADER_BYTES:]
                self.raw_buffer.extend(payload)
                self.total_payload_bytes += len(payload)

                while len(self.raw_buffer) >= FRAME_BYTES:
                    frame_raw = self.raw_buffer[:FRAME_BYTES]
                    del self.raw_buffer[:FRAME_BYTES]

                    self.frame_count += 1

                    adc_cube = parse_one_frame_iwr6843_dca1000_complex(frame_raw)

                    # AIC 必須在 Range FFT / Doppler FFT 之前做，
                    # 也就是先對 raw ADC IQ 做背景扣除，再轉成 complex RDI。
                    current_aic_enabled = self.controls.snapshot()["use_aic"]
                    if current_aic_enabled != self._last_aic_enabled:
                        self.aic.enabled = bool(current_aic_enabled)
                        self.aic.reset()
                        self._last_aic_enabled = bool(current_aic_enabled)
                        print(f"[AIC] switched {'ON' if current_aic_enabled else 'OFF'}; AIC reference reset.")
                    else:
                        self.aic.enabled = bool(current_aic_enabled)

                    adc_cube_for_rdi = self.aic.apply(adc_cube)

                    rdi_complex = compute_rdi_per_rx(adc_cube_for_rdi)
                    self.last_model_input = crop_rdi_for_model(rdi_complex)

                    # 即時顯示用：只保存最新一幀的 complex RDI。
                    with self.rdi_lock:
                        self.latest_rdi_complex = rdi_complex.copy()
                        self.latest_rdi_frame_id = self.frame_count

                    if self.frame_count % DISPLAY_EVERY_N_FRAMES == 0:
                        print(
                            f"[FRAME] {self.frame_count}, "
                            f"adc_cube={adc_cube.shape}, "
                            f"rdi={rdi_complex.shape}, "
                            f"model={self.last_model_input.shape}, "
                            f"aic={'ON' if self.aic.enabled else 'OFF'}, aic_W={self.aic.W}, "
                            f"remain_buffer={len(self.raw_buffer)}, "
                            f"lost={self.lost_packets}"
                        )

                now = time.time()
                if now - last_log >= 1.0:
                    elapsed = now - t0
                    mb = self.total_payload_bytes / 1024 / 1024
                    rate = mb / max(elapsed, 1e-6)

                    print(
                        f"[STAT] packets={self.total_packets}, "
                        f"frames={self.frame_count}, "
                        f"payload={mb:.2f} MB, "
                        f"rate={rate:.2f} MB/s, "
                        f"lost={self.lost_packets}"
                    )

                    last_log = now

        finally:
            sock.close()
            print("[UDP] receiver stopped.")


# ============================================================
# Main
# ============================================================

def main():
    args, _ = build_arg_parser().parse_known_args()
    record_frames = max(1, int(args.record_frames))
    record_video_fps = max(0.1, float(args.record_video_fps))

    radar_ser = None
    viewer = None
    camera_recorder = None
    controls = RuntimeRDIControls()
    receiver = DCA1000UDPReceiver(controls=controls)

    print("====================================")
    print("Python-only DCA1000 continuous raw capture")
    print("====================================")
    print(f"Radar cfg : {RADAR_CFG}")
    print(f"DCA JSON  : {DCA_JSON}")
    print(f"DCA CLI   : {DCA_CONTROL_EXE}")
    print("Display   : 2 panels only (RDI + two-RDI azimuth/PHD)")
    print("Controls  : removed; using code defaults only")
    print(
        f"Recording : press r to input frame count and start recording "
        f"(default {record_frames}), s to stop/save, q/ESC to close"
    )
    print(f"Record FPS: {record_video_fps:g}")
    print("====================================")

    latest_record_frame = None
    latest_record_frame_id = -1

    try:
        # ----------------------------------------------------
        # 1. Open radar CLI and send cfg without sensorStart
        # ----------------------------------------------------
        radar_ser = open_radar_cli()
        send_radar_cfg_without_sensor_start(radar_ser)

        # ----------------------------------------------------
        # 2. Configure DCA1000, but do NOT call CLI start_record
        # ----------------------------------------------------
        run_dca_command("fpga", check=True)
        run_dca_command("record", check=True)

        # ----------------------------------------------------
        # 3. Start Python UDP receiver first
        # ----------------------------------------------------
        receiver.start()
        time.sleep(1.0)

        # ----------------------------------------------------
        # 4. Start DCA1000 record by direct UDP command
        # ----------------------------------------------------
        dca_start_record_direct()
        time.sleep(0.2)

        # ----------------------------------------------------
        # 5. Start radar sensor
        # ----------------------------------------------------
        radar_sensor_start(radar_ser)

        print("====================================")
        print("Continuous raw capture started.")
        print("Press Ctrl+C to stop.")
        print("====================================")

        if SHOW_RDI_REALTIME:
            if cv2 is None:
                print("[WARN] OpenCV is not available, realtime RDI display disabled.")
                print("       Install with: pip install opencv-python")
            else:
                viewer = RealtimeRDIViewer(
                    num_rx=NUM_RX,
                    range_bins=SAVE_RANGE_BINS,
                    doppler_bins=SAVE_DOPPLER_BINS,
                    controls=controls,
                )

        camera_recorder = AprilTagCameraRecorder(
            enabled=ENABLE_APRILTAG_CAMERA,
            record_video_fps=record_video_fps,
        )

        while True:
            # Camera 盡量每圈更新，讓 AprilTag label 保持最新。
            cam_key = -1
            camera_panel = None
            if camera_recorder is not None:
                cam_key = camera_recorder.update_camera()
                camera_panel = camera_recorder.get_display_frame()

            radar_key = -1
            if viewer is not None and not viewer.closed:
                latest_rdi, latest_frame_id = receiver.get_latest_rdi()
                if latest_rdi is not None:
                    maybe_frame = viewer.update(latest_rdi, latest_frame_id, camera_panel=camera_panel)
                    radar_key = viewer.last_key
                    if maybe_frame is not None:
                        latest_record_frame = maybe_frame
                        latest_record_frame_id = latest_frame_id
                        if camera_recorder is not None:
                            camera_recorder.record_radar_frame(latest_frame_id, maybe_frame)
                else:
                    # OpenCV 需要 waitKey 才能處理視窗事件。
                    if cv2 is not None:
                        radar_key = cv2.waitKey(1) & 0xFF
                    time.sleep(0.005)
            else:
                if cv2 is not None:
                    radar_key = cv2.waitKey(1) & 0xFF
                time.sleep(0.05)

            key_candidates = [cam_key, radar_key]
            if any(k in (27, ord('q')) for k in key_candidates):
                print("[STOP] q/ESC received.")
                break
            if any(k == ord('r') for k in key_candidates):
                if camera_recorder is not None:
                    requested_frames = ask_record_frame_count(record_frames)
                    if requested_frames is not None:
                        record_frames = requested_frames
                        camera_recorder.start_recording(record_frames, latest_record_frame)
            if any(k == ord('s') for k in key_candidates):
                if camera_recorder is not None:
                    camera_recorder.finish_recording()

            if viewer is not None and viewer.closed:
                break

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C received.")

    except Exception as e:
        print(f"\n[ERROR] {e}")

    finally:
        print("====================================")
        print("Cleanup")
        print("====================================")

        if radar_ser is not None:
            radar_sensor_stop(radar_ser)

        dca_stop_record_direct()

        receiver.stop()

        if camera_recorder is not None:
            camera_recorder.close()

        if viewer is not None:
            viewer.close()

        if SAVE_LAST_MODEL_INPUT and receiver.last_model_input is not None:
            np.save(LAST_MODEL_INPUT_PATH, receiver.last_model_input)
            print(f"[SAVE] {LAST_MODEL_INPUT_PATH.resolve()}")

        if radar_ser is not None:
            try:
                radar_ser.close()
            except Exception:
                pass

        print("====================================")
        print("Finished")
        print(f"Packets       : {receiver.total_packets}")
        print(f"Frames        : {receiver.frame_count}")
        print(f"Payload bytes : {receiver.total_payload_bytes:,}")
        print(f"Lost packets  : {receiver.lost_packets}")
        print("====================================")


if __name__ == "__main__":
    main()
