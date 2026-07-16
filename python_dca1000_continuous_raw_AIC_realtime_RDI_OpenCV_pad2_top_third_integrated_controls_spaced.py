import os
import socket
import struct
import time
import subprocess
import threading
from pathlib import Path

import serial
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

CLI_PORT = "COM9"
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
RDI_DISPLAY_RANGE_START_BIN = RANGE_FFT_SIZE * 2 // 3  # inclusive; 64 -> 42
RDI_DISPLAY_RANGE_END_BIN = RANGE_FFT_SIZE             # exclusive; 64 -> 顯示 42~63

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
RDI_WINDOW_NAME = "Realtime per-RX RDI with integrated controls"

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
    GUI/receiver 共用的即時控制狀態。

    - use_aic：receiver thread 會讀取，決定 raw ADC 是否套用 AIC。
    - use_complex_bg：viewer 會讀取，決定 GUI 是否顯示 complex background subtraction 後的 RDI。
    - fixed / complex bg 的 vmin/vmax：viewer 會讀取，用於即時調整 colormap 色階。
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.use_aic = bool(AIC_ON)
        self.use_complex_bg = bool(RDI_USE_COMPLEX_BG_SUBTRACTION_DISPLAY)
        self.fixed_vmin_db = float(RDI_FIXED_VMIN_DB)
        self.fixed_vmax_db = float(RDI_FIXED_VMAX_DB)
        self.complex_vmin_db = float(RDI_COMPLEX_BG_VMIN_DB)
        self.complex_vmax_db = float(RDI_COMPLEX_BG_VMAX_DB)

    @staticmethod
    def _sanitize_pair(vmin, vmax):
        vmin = float(vmin)
        vmax = float(vmax)
        if vmax <= vmin:
            vmax = vmin + 1.0
        return vmin, vmax

    def snapshot(self):
        with self.lock:
            fixed_vmin, fixed_vmax = self._sanitize_pair(self.fixed_vmin_db, self.fixed_vmax_db)
            complex_vmin, complex_vmax = self._sanitize_pair(self.complex_vmin_db, self.complex_vmax_db)
            return {
                "use_aic": bool(self.use_aic),
                "use_complex_bg": bool(self.use_complex_bg),
                "fixed_vmin_db": fixed_vmin,
                "fixed_vmax_db": fixed_vmax,
                "complex_vmin_db": complex_vmin,
                "complex_vmax_db": complex_vmax,
            }

    def set_use_aic(self, value):
        with self.lock:
            self.use_aic = bool(value)

    def set_use_complex_bg(self, value):
        with self.lock:
            self.use_complex_bg = bool(value)

    def set_fixed_vmin(self, value):
        with self.lock:
            self.fixed_vmin_db = float(value)

    def set_fixed_vmax(self, value):
        with self.lock:
            self.fixed_vmax_db = float(value)

    def set_complex_vmin(self, value):
        with self.lock:
            self.complex_vmin_db = float(value)

    def set_complex_vmax(self, value):
        with self.lock:
            self.complex_vmax_db = float(value)

class RealtimeRDIViewer:
    """
    OpenCV 即時顯示所有 RX 的最後一幀 RDI，並額外顯示 range profile。

    這版的目的不是只看顏色，而是讓「距離位置」更明顯：
        1. 每張 RDI 右側有 1D range profile。
        2. RDI 上會畫出目前 range peak 的水平線。
        3. 文字顯示 range_peak 與 rdi_peak。
    """

    def __init__(self, num_rx, range_bins, doppler_bins, controls=None):
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not available. Please install opencv-python or set SHOW_RDI_REALTIME = False."
            )

        self.num_rx = int(num_rx)
        self.controls = controls if controls is not None else RuntimeRDIControls()

        # full_range_bins 是 RDI 計算結果的完整 range bins。
        # display_range_start/end 只控制 GUI 要顯示哪一段 range bins，
        # 不會影響 receiver 裡的 rdi_complex 或 last_model_input。
        self.full_range_bins = int(range_bins)
        self.display_range_start, self.display_range_end = self._get_display_range_slice(self.full_range_bins)
        self.range_bins = int(self.display_range_end - self.display_range_start)
        self.doppler_bins = int(doppler_bins)

        self.last_frame_id = -1
        self.last_update_time = 0.0
        self.closed = False

        self.tile_w = int(RDI_TILE_WIDTH)
        self.tile_h = int(RDI_TILE_HEIGHT)
        self.profile_w = int(RDI_PROFILE_WIDTH) if RDI_DRAW_RANGE_PROFILE else 0
        self.profile_gap = int(RDI_PROFILE_GAP) if RDI_DRAW_RANGE_PROFILE else 0
        self.heatmap_w = max(160, self.tile_w - self.profile_w - self.profile_gap)
        self.gap = int(RDI_TILE_GAP)
        self.border = int(RDI_TILE_BORDER)
        self.title_bar_h = int(RDI_TITLE_BAR_HEIGHT)

        # RDI 主視窗加寬，讓下方控制項文字可以完整顯示。
        self.grid_w = self.tile_w * 2 + self.gap
        self.grid_h = self.tile_h * 2 + self.gap
        self.canvas_w = max(int(RDI_INTEGRATED_WINDOW_MIN_WIDTH), self.border * 2 + self.grid_w)
        self.control_h = int(RDI_INTEGRATED_CONTROL_HEIGHT)
        self.canvas_h = self.title_bar_h + self.border * 2 + self.grid_h + self.control_h

        # RDI 2x2 tiles 置中顯示；控制項放在 tiles 下方。
        self.grid_x0 = (self.canvas_w - self.grid_w) // 2
        self.grid_y0 = self.title_bar_h + self.border
        self.control_y0 = self.grid_y0 + self.grid_h + 22

        # 滑桿 / 勾選框的滑鼠互動狀態。
        self.ui_items = {}
        self.dragging_slider = None

        cv2.namedWindow(RDI_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RDI_WINDOW_NAME, self.canvas_w, self.canvas_h)
        cv2.setMouseCallback(RDI_WINDOW_NAME, self._on_mouse_event)

        # 背景校正只用在 GUI 顯示端。receiver 仍然保存原本的 complex RDI。
        # magnitude-ratio 顯示用背景。
        self.bg_accum_mag = None
        self.bg_mag = None
        self.bg_count = 0
        self.bg_ready = False

        # complex background subtraction 顯示用背景。
        self.bg_accum_complex = None
        self.bg_complex = None
        self.complex_bg_count = 0
        self.complex_bg_ready = False

        # 建立控制面板：0/1 trackbar 模擬勾選；dB trackbar 即時調整色階。
        self._create_control_panel()

    def _create_control_panel(self):
        """保留函式名稱，但這版不另外建立控制視窗。控制項直接畫在主 RDI 視窗內。"""
        print("[CTRL] Integrated controls are shown inside the RDI window.")

    def _toggle_aic(self):
        old = self.controls.snapshot()["use_aic"]
        new_value = not old
        self.controls.set_use_aic(new_value)
        print(f"[CTRL] AIC processing {'ON' if new_value else 'OFF'}")

    def _toggle_complex_bg(self):
        old = self.controls.snapshot()["use_complex_bg"]
        new_value = not old
        self.controls.set_use_complex_bg(new_value)
        print(f"[CTRL] Complex background subtraction display {'ON' if new_value else 'OFF'}")
        if new_value:
            # 重新開啟 complex bg display 時重新校正背景，避免使用舊背景。
            self.reset_background()

    def _set_slider_value_from_x(self, name, x):
        item = self.ui_items.get(name)
        if item is None:
            return
        x1, x2 = item["track"]
        if x2 <= x1:
            return
        ratio = (float(x) - float(x1)) / float(x2 - x1)
        ratio = max(0.0, min(1.0, ratio))
        value = RDI_SLIDER_DB_MIN + ratio * (RDI_SLIDER_DB_MAX - RDI_SLIDER_DB_MIN)
        value = int(round(value))

        if name == "fixed_vmin":
            self.controls.set_fixed_vmin(value)
        elif name == "fixed_vmax":
            self.controls.set_fixed_vmax(value)
        elif name == "complex_vmin":
            self.controls.set_complex_vmin(value)
        elif name == "complex_vmax":
            self.controls.set_complex_vmax(value)

    def _on_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, item in self.ui_items.items():
                kind = item.get("kind")
                if kind == "checkbox":
                    x1, y1, x2, y2 = item["bbox"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        if name == "use_aic":
                            self._toggle_aic()
                        elif name == "use_complex_bg":
                            self._toggle_complex_bg()
                        return
                elif kind == "slider":
                    x1, y1, x2, y2 = item["bbox"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.dragging_slider = name
                        self._set_slider_value_from_x(name, x)
                        return

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_slider is not None:
                self._set_slider_value_from_x(self.dragging_slider, x)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_slider is not None:
                self._set_slider_value_from_x(self.dragging_slider, x)
            self.dragging_slider = None

    def _draw_checkbox(self, canvas, name, label, enabled, x, y):
        box_size = 18
        box_x1 = x
        box_y1 = y - box_size + 3
        box_x2 = box_x1 + box_size
        box_y2 = box_y1 + box_size
        self.ui_items[name] = {
            "kind": "checkbox",
            "bbox": (box_x1 - 8, box_y1 - 8, self.canvas_w - 24, box_y2 + 8),
        }

        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (230, 230, 230), 1, cv2.LINE_AA)
        if enabled:
            cv2.line(canvas, (box_x1 + 4, box_y1 + 10), (box_x1 + 8, box_y1 + 14), (80, 255, 120), 2, cv2.LINE_AA)
            cv2.line(canvas, (box_x1 + 8, box_y1 + 14), (box_x1 + 15, box_y1 + 4), (80, 255, 120), 2, cv2.LINE_AA)

        state_text = "ON" if enabled else "OFF"
        state_color = (80, 255, 120) if enabled else (170, 170, 170)
        cv2.putText(canvas, label, (x + 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, state_text, (self.canvas_w - 82, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 1, cv2.LINE_AA)

    def _draw_slider(self, canvas, name, label, value, x, y):
        value = float(value)

        # 這版把「文字」和「滑桿」分成上下兩列，避免長標籤和數值擠在一起。
        label_y = y
        track_x1 = x + 40
        track_x2 = self.canvas_w - 95
        track_y = y + 26
        value_text = f"{value:.0f} dB"
        value_x = self.canvas_w - 205

        # 標籤列：左邊顯示完整說明，右邊顯示目前數值。
        cv2.putText(canvas, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, value_text, (value_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 235, 160), 1, cv2.LINE_AA)

        # 滑桿列：獨立放在標籤下方，所以不會壓到文字。
        cv2.line(canvas, (track_x1, track_y), (track_x2, track_y), (135, 135, 135), 4, cv2.LINE_AA)
        ratio = (value - RDI_SLIDER_DB_MIN) / max(RDI_SLIDER_DB_MAX - RDI_SLIDER_DB_MIN, 1)
        ratio = max(0.0, min(1.0, ratio))
        knob_x = int(round(track_x1 + ratio * (track_x2 - track_x1)))
        cv2.line(canvas, (track_x1, track_y), (knob_x, track_y), (90, 180, 255), 4, cv2.LINE_AA)
        cv2.circle(canvas, (knob_x, track_y), 8, (70, 170, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (knob_x, track_y), 8, (255, 255, 255), 1, cv2.LINE_AA)

        self.ui_items[name] = {
            "kind": "slider",
            "track": (track_x1, track_x2),
            # 讓使用者點在文字附近或滑桿附近都可以拖曳。
            "bbox": (track_x1 - 24, label_y - 20, track_x2 + 24, track_y + 18),
        }

    def _draw_integrated_controls(self, canvas, vmin, vmax, scale_text):
        ctrl = self.controls.snapshot()
        self.ui_items = {}

        panel_x1 = 14
        panel_y1 = self.control_y0 - 18
        panel_x2 = self.canvas_w - 14
        panel_y2 = self.canvas_h - 14
        cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (34, 34, 34), -1)
        cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (90, 90, 90), 1, cv2.LINE_AA)

        x = 28
        y = self.control_y0 + 8
        cv2.putText(canvas, "Runtime Display Controls", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"Current display scale: {scale_text} | active vmin/vmax = {vmin:.0f}/{vmax:.0f} dB",
            (x + 300, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (210, 230, 255),
            1,
            cv2.LINE_AA,
        )

        y += 34
        self._draw_checkbox(canvas, "use_aic", "Enable AIC processing before Range/Doppler FFT", ctrl["use_aic"], x, y)
        y += 32
        self._draw_checkbox(canvas, "use_complex_bg", "Enable complex background subtraction for display", ctrl["use_complex_bg"], x, y)

        y += 50
        self._draw_slider(canvas, "fixed_vmin", "AIC-only fixed display minimum dB", ctrl["fixed_vmin_db"], x, y)
        y += 58
        self._draw_slider(canvas, "fixed_vmax", "AIC-only fixed display maximum dB", ctrl["fixed_vmax_db"], x, y)
        y += 58
        self._draw_slider(canvas, "complex_vmin", "Complex background-subtracted display minimum dB", ctrl["complex_vmin_db"], x, y)
        y += 58
        self._draw_slider(canvas, "complex_vmax", "Complex background-subtracted display maximum dB", ctrl["complex_vmax_db"], x, y)

        cv2.putText(
            canvas,
            "Mouse: click checkboxes or drag sliders | Keyboard: b = recalibrate background, q / ESC = close",
            (x, self.canvas_h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    def _get_display_range_slice(self, full_range_bins):
        """
        決定 GUI 要顯示哪一段 range bins。

        注意：這只影響顯示，不改變 RDI 計算本身。
        目前 RDI_DISPLAY_ORIGIN="lower" 時，畫面上方對應較大的 range bin。
        """
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

    def _crop_display_range(self, display_db):
        """
        display_db shape = (rx, full_range_bins, doppler_bins)
        return shape = (rx, displayed_range_bins, doppler_bins)
        """
        return display_db[:, self.display_range_start:self.display_range_end, :]

    def reset_background(self):
        self.bg_accum_mag = None
        self.bg_mag = None
        self.bg_count = 0
        self.bg_ready = False

        self.bg_accum_complex = None
        self.bg_complex = None
        self.complex_bg_count = 0
        self.complex_bg_ready = False

        print("[VIEW] Background calibration reset.")

    def _prepare_display_db(self, rdi_complex):
        """
        回傳實際拿來顯示的 dB 圖與顯示尺度。

        顯示模式優先順序：
            1. complex background subtraction display
               display_db = 20log10(abs(rdi_complex - bg_complex) + eps)
            2. magnitude-ratio background delta display
               display_db = 20log10((abs(rdi_complex)+eps) / bg_mag)
            3. 原始 RDI magnitude display
               display_db = 20log10(abs(rdi_complex)+eps)
        """
        ctrl = self.controls.snapshot()

        if ctrl["use_complex_bg"]:
            if self.bg_accum_complex is None:
                self.bg_accum_complex = np.zeros_like(rdi_complex, dtype=np.complex128)
                self.bg_complex = np.zeros_like(rdi_complex, dtype=np.complex64)
                self.complex_bg_count = 0
                self.complex_bg_ready = False

            if self.complex_bg_count < int(RDI_COMPLEX_BG_WARMUP_FRAMES):
                self.bg_accum_complex += rdi_complex.astype(np.complex128)
                self.complex_bg_count += 1
                self.bg_complex = (
                    self.bg_accum_complex / max(self.complex_bg_count, 1)
                ).astype(np.complex64)
                self.complex_bg_ready = self.complex_bg_count >= int(RDI_COMPLEX_BG_WARMUP_FRAMES)

                display_db = np.zeros_like(np.abs(rdi_complex), dtype=np.float32)
                return display_db, float(ctrl["complex_vmin_db"]), float(ctrl["complex_vmax_db"]), (
                    f"calibrating complex bg {self.complex_bg_count}/{RDI_COMPLEX_BG_WARMUP_FRAMES}"
                )

            residual = rdi_complex.astype(np.complex64) - self.bg_complex
            display_db = 20.0 * np.log10(np.abs(residual).astype(np.float32) + RDI_DISPLAY_EPS)
            display_db = np.nan_to_num(
                display_db,
                nan=ctrl["complex_vmin_db"],
                posinf=ctrl["complex_vmax_db"],
                neginf=ctrl["complex_vmin_db"],
            )

            return display_db.astype(np.float32), float(ctrl["complex_vmin_db"]), float(ctrl["complex_vmax_db"]), (
                f"complex-bg-sub {ctrl['complex_vmin_db']:.0f}~{ctrl['complex_vmax_db']:.0f} dB | b=reset bg"
            )

        mag = np.abs(rdi_complex).astype(np.float32)

        if RDI_USE_BACKGROUND_DELTA_DISPLAY:
            if self.bg_accum_mag is None:
                self.bg_accum_mag = np.zeros_like(mag, dtype=np.float64)
                self.bg_mag = np.zeros_like(mag, dtype=np.float32)
                self.bg_count = 0
                self.bg_ready = False

            if self.bg_count < int(RDI_BACKGROUND_WARMUP_FRAMES):
                self.bg_accum_mag += mag.astype(np.float64)
                self.bg_count += 1
                self.bg_mag = (self.bg_accum_mag / max(self.bg_count, 1)).astype(np.float32)
                self.bg_ready = self.bg_count >= int(RDI_BACKGROUND_WARMUP_FRAMES)

                display_db = np.zeros_like(mag, dtype=np.float32)
                return display_db, float(RDI_DELTA_VMIN_DB), float(RDI_DELTA_VMAX_DB), (
                    f"calibrating bg {self.bg_count}/{RDI_BACKGROUND_WARMUP_FRAMES}"
                )

            bg = np.maximum(self.bg_mag, RDI_DISPLAY_EPS)
            display_db = 20.0 * np.log10((mag + RDI_DISPLAY_EPS) / bg)
            display_db = np.nan_to_num(
                display_db,
                nan=0.0,
                posinf=RDI_DELTA_VMAX_DB,
                neginf=RDI_DELTA_VMIN_DB,
            )

            if RDI_CLIP_NEGATIVE_DELTA_DB:
                display_db = np.maximum(display_db, 0.0)

            return display_db.astype(np.float32), float(RDI_DELTA_VMIN_DB), float(RDI_DELTA_VMAX_DB), (
                f"bg-delta {RDI_DELTA_VMIN_DB:.0f}~{RDI_DELTA_VMAX_DB:.0f} dB | b=reset bg"
            )

        rdi_db = rdi_complex_to_db(rdi_complex)
        finite = np.isfinite(rdi_db)
        if not np.any(finite):
            return rdi_db, 0.0, 1.0, "no finite data"

        if RDI_USE_FIXED_DB_SCALE:
            vmin = float(ctrl["fixed_vmin_db"])
            vmax = float(ctrl["fixed_vmax_db"])
            scale_text = f"fixed {vmin:.0f}~{vmax:.0f} dB"
        else:
            vmax = float(np.max(rdi_db[finite]))
            vmin = vmax - float(RDI_DISPLAY_DYNAMIC_RANGE_DB)
            scale_text = f"relative {RDI_DISPLAY_DYNAMIC_RANGE_DB:.0f} dB"

        return rdi_db.astype(np.float32), vmin, vmax, scale_text

    def _normalize_to_uint8(self, img_db, vmin, vmax):
        img = np.asarray(img_db, dtype=np.float32)
        img = np.nan_to_num(img, nan=vmin, posinf=vmax, neginf=vmin)
        img = np.clip(img, vmin, vmax)

        denom = max(vmax - vmin, 1e-6)
        img_u8 = ((img - vmin) / denom * 255.0).astype(np.uint8)
        return img_u8

    def _range_profile(self, img_db):
        """
        從 RDI display_db 產生 1D range profile。
        output shape = (range_bins,)
        """
        work = np.asarray(img_db, dtype=np.float32).copy()
        work[~np.isfinite(work)] = -np.inf

        if RDI_RANGE_IGNORE_CENTER_DOPPLER_FOR_PROFILE:
            center = work.shape[1] // 2
            lo = max(0, center - RDI_RANGE_CENTER_HALF_WIDTH)
            hi = min(work.shape[1], center + RDI_RANGE_CENTER_HALF_WIDTH + 1)
            work[:, lo:hi] = -np.inf

        if RDI_RANGE_PROFILE_MODE.lower() == "mean":
            finite = np.isfinite(work)
            safe = np.where(finite, work, np.nan)
            profile = np.nanmean(safe, axis=1)
            profile = np.nan_to_num(profile, nan=-np.inf)
        else:
            profile = np.max(work, axis=1)

        ignore_n = int(RDI_IGNORE_NEAR_RANGE_BINS)
        if ignore_n > 0:
            profile[:ignore_n] = -np.inf

        return profile.astype(np.float32)

    def _find_rdi_peak(self, img_db):
        work = np.asarray(img_db, dtype=np.float32).copy()
        work[~np.isfinite(work)] = -np.inf

        if RDI_PEAK_IGNORE_CENTER_DOPPLER:
            center = work.shape[1] // 2
            lo = max(0, center - RDI_PEAK_CENTER_HALF_WIDTH)
            hi = min(work.shape[1], center + RDI_PEAK_CENTER_HALF_WIDTH + 1)
            work[:, lo:hi] = -np.inf

        idx = int(np.nanargmax(work))
        r_bin, d_bin = np.unravel_index(idx, work.shape)
        return int(r_bin), int(d_bin), float(img_db[r_bin, d_bin])

    def _make_profile_panel(self, profile, range_peak, vmin, vmax):
        panel = np.zeros((self.tile_h, self.profile_w, 3), dtype=np.uint8)
        if self.profile_w <= 0:
            return panel

        cv2.putText(
            panel,
            "range",
            (5, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        profile = np.asarray(profile, dtype=np.float32)
        profile = np.nan_to_num(profile, nan=vmin, posinf=vmax, neginf=vmin)
        profile = np.clip(profile, vmin, vmax)
        denom = max(vmax - vmin, 1e-6)
        norm = (profile - vmin) / denom

        for r in range(self.range_bins):
            if RDI_DISPLAY_ORIGIN.lower() == "lower":
                y_center = int((self.range_bins - 1 - r + 0.5) * self.tile_h / self.range_bins)
            else:
                y_center = int((r + 0.5) * self.tile_h / self.range_bins)

            bin_h = max(2, int(self.tile_h / self.range_bins) - 2)
            y1 = max(0, y_center - bin_h // 2)
            y2 = min(self.tile_h - 1, y_center + bin_h // 2)
            bar_w = int(norm[r] * (self.profile_w - 18))

            color = (80, 220, 255) if r != range_peak else (0, 255, 255)
            cv2.rectangle(panel, (8, y1), (8 + bar_w, y2), color, -1)

        # range peak 水平線
        if RDI_DISPLAY_ORIGIN.lower() == "lower":
            py = int((self.range_bins - 1 - range_peak + 0.5) * self.tile_h / self.range_bins)
        else:
            py = int((range_peak + 0.5) * self.tile_h / self.range_bins)
        cv2.line(panel, (0, py), (self.profile_w - 1, py), (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(
            panel,
            f"r={range_peak}",
            (5, self.tile_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return panel

    def _make_tile(self, img_db, rx_idx, vmin, vmax):
        """
        單一 RX RDI -> heatmap tile。
        img_db shape = (displayed_range_bins, doppler_bins)
        """
        rdi_peak_r, rdi_peak_d, rdi_peak_db = self._find_rdi_peak(img_db)
        rdi_peak_r_original = rdi_peak_r + self.display_range_start

        # Range profile / range line 已關閉時，不再額外計算或顯示距離長條。
        if RDI_DRAW_RANGE_LINE or RDI_DRAW_RANGE_PROFILE:
            profile = self._range_profile(img_db)
            range_peak = int(np.nanargmax(profile)) if np.any(np.isfinite(profile)) else rdi_peak_r
            range_peak_db = float(profile[range_peak]) if np.isfinite(profile[range_peak]) else rdi_peak_db
        else:
            profile = None
            range_peak = rdi_peak_r
            range_peak_db = rdi_peak_db

        img_show = np.asarray(img_db, dtype=np.float32).copy()
        if RDI_MASK_LOW_DB:
            img_show[img_show < RDI_MASK_THRESHOLD_DB] = vmin

        if RDI_DISPLAY_ORIGIN.lower() == "lower":
            img_show = np.flipud(img_show)
            rdi_peak_y_bin = self.range_bins - 1 - rdi_peak_r
            range_peak_y_bin = self.range_bins - 1 - range_peak
        else:
            rdi_peak_y_bin = rdi_peak_r
            range_peak_y_bin = range_peak

        img_u8 = self._normalize_to_uint8(img_show, vmin, vmax)
        img_u8 = cv2.resize(
            img_u8,
            (self.heatmap_w, self.tile_h),
            interpolation=cv2.INTER_NEAREST,
        )
        heatmap = cv2.applyColorMap(img_u8, RDI_COLORMAP)

        # range peak 水平線，比單一 cross 更容易看距離位置。
        if RDI_DRAW_RANGE_LINE:
            py_line = int((range_peak_y_bin + 0.5) * self.tile_h / max(self.range_bins, 1))
            cv2.line(heatmap, (0, py_line), (self.heatmap_w - 1, py_line), (0, 255, 255), 2, cv2.LINE_AA)

        if RDI_DRAW_PEAK_MARKER:
            px = int((rdi_peak_d + 0.5) * self.heatmap_w / max(self.doppler_bins, 1))
            py = int((rdi_peak_y_bin + 0.5) * self.tile_h / max(self.range_bins, 1))
            cv2.drawMarker(
                heatmap,
                (px, py),
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        cv2.putText(
            heatmap,
            f"RX{rx_idx + 1}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            heatmap,
            f"peak r={rdi_peak_r_original}, d={rdi_peak_d}, {rdi_peak_db:.1f} dB",
            (12, self.tile_h - 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
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
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if RDI_DRAW_RANGE_PROFILE:
            gap_img = np.zeros((self.tile_h, self.profile_gap, 3), dtype=np.uint8)
            panel = self._make_profile_panel(profile, range_peak, vmin, vmax)
            tile = np.concatenate([heatmap, gap_img, panel], axis=1)
        else:
            tile = heatmap

        # 保證 tile 寬度符合 canvas。
        if tile.shape[1] != self.tile_w:
            tile = cv2.resize(tile, (self.tile_w, self.tile_h), interpolation=cv2.INTER_NEAREST)
        return tile

    def update(self, rdi_complex, frame_id, force=False):
        if self.closed:
            return

        now = time.time()
        if (not force) and (now - self.last_update_time < RDI_DISPLAY_INTERVAL_SEC):
            return

        if frame_id == self.last_frame_id:
            return

        display_db, vmin, vmax, scale_text = self._prepare_display_db(rdi_complex)
        display_db = self._crop_display_range(display_db)
        finite = np.isfinite(display_db)
        if not np.any(finite):
            return

        tiles = []
        for rx in range(self.num_rx):
            tiles.append(self._make_tile(display_db[rx], rx, vmin, vmax))

        blank = np.zeros((self.tile_h, self.tile_w, 3), dtype=np.uint8)
        while len(tiles) < 4:
            tiles.append(blank.copy())

        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        ctrl = self.controls.snapshot()
        title = (
            f"Realtime per-RX RDI | frame={frame_id} | "
            f"AIC={'ON' if ctrl['use_aic'] else 'OFF'} | "
            f"ComplexBG={'ON' if ctrl['use_complex_bg'] else 'OFF'} | "
            f"SCR={'ON' if STATIC_CLUTTER_REMOVAL else 'OFF'} | {scale_text} | "
            f"range shown={self.display_range_start}-{self.display_range_end - 1}/{self.full_range_bins}, "
            f"doppler={self.doppler_bins} | cube={DCA1000_CUBE_ORDER} | b=reset bg | q/ESC close"
        )
        cv2.putText(
            canvas,
            title,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y0 = self.grid_y0
        x0 = self.grid_x0
        positions = [
            (x0, y0),
            (x0 + self.tile_w + self.gap, y0),
            (x0, y0 + self.tile_h + self.gap),
            (x0 + self.tile_w + self.gap, y0 + self.tile_h + self.gap),
        ]

        for tile, (x, y) in zip(tiles[:4], positions):
            canvas[y:y + self.tile_h, x:x + self.tile_w] = tile
            cv2.rectangle(
                canvas,
                (x - 1, y - 1),
                (x + self.tile_w, y + self.tile_h),
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )

        self._draw_integrated_controls(canvas, vmin, vmax, scale_text)

        cv2.imshow(RDI_WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            self.closed = True
            cv2.destroyWindow(RDI_WINDOW_NAME)
        elif key == RDI_RESET_BACKGROUND_KEY:
            self.reset_background()

        self.last_frame_id = frame_id
        self.last_update_time = now

    def close(self):
        self.closed = True
        if cv2 is not None:
            try:
                cv2.destroyWindow(RDI_WINDOW_NAME)
            except Exception:
                pass

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
    radar_ser = None
    viewer = None
    controls = RuntimeRDIControls()
    receiver = DCA1000UDPReceiver(controls=controls)

    print("====================================")
    print("Python-only DCA1000 continuous raw capture")
    print("====================================")
    print(f"Radar cfg : {RADAR_CFG}")
    print(f"DCA JSON  : {DCA_JSON}")
    print(f"DCA CLI   : {DCA_CONTROL_EXE}")
    print("====================================")

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

        while True:
            if viewer is not None and not viewer.closed:
                latest_rdi, latest_frame_id = receiver.get_latest_rdi()
                if latest_rdi is not None:
                    viewer.update(latest_rdi, latest_frame_id)
                else:
                    # OpenCV 需要 waitKey 才能處理視窗事件。
                    cv2.waitKey(1)
                    time.sleep(0.005)
            else:
                time.sleep(0.05)

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