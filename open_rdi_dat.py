import struct
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 基本設定
# ============================================================

file_path = r"C:\Users\mc2\Desktop\degree_trainer\xwr68xx_processed_stream_2026_04_24T07_29_53_534.dat"

MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# 你的資料目前 TLV type 5 size = 4096 個 uint16
# 所以 128 * 32 = 4096
RANGE_BINS = 256
DOPPLER_BINS = 16

# ============================================================
# 2. 顯示尺度設定
# ============================================================
# 這兩個值要依照你的 radar cfg 正確計算。
# 下面先放近似值，讓畫面先能像 Demo Visualizer 那樣顯示實際座標。
#
# 如果你只是想先看圖，可以先不用改。
# 如果你要論文或實驗數據精準，之後要從 cfg 算出正確解析度。

RANGE_RESOLUTION_M = 10.0 / RANGE_BINS
# 讓 x 軸大約顯示 0~10 meters
# 若你的設定不是 10m 最大距離，請改這個值。

DOPPLER_RESOLUTION_MPS = 0.8 / (DOPPLER_BINS / 2)
# 讓 y 軸大約顯示 -0.8 ~ 0.8 m/s
# 若你的設定不是這個速度範圍，請改這個值。

MAX_RANGE_M = RANGE_BINS * RANGE_RESOLUTION_M
MAX_DOPPLER_MPS = (DOPPLER_BINS / 2) * DOPPLER_RESOLUTION_MPS


# ============================================================
# 3. 播放設定
# ============================================================

FPS = 10
LOOP_PLAYBACK = True

# 是否固定顏色範圍
# True：整段影片使用同一組顏色範圍，比較不會閃爍
# False：每一幀自動調整顏色範圍，局部變化比較明顯
FIX_COLOR_SCALE = True

# 顏色範圍百分位數，用來避免極端值影響顯示
COLOR_PERCENTILE_LOW = 5
COLOR_PERCENTILE_HIGH = 99


# ============================================================
# 4. 是否印出指定 frame 的數值
# ============================================================

PRINT_TARGET_FRAME = True
TARGET_FRAME = 10
PRINT_FULL_MATRIX = False


# ============================================================
# 5. RDI 顯示轉換函式
# ============================================================

def convert_rdi_to_visualizer_style(rdi_raw):
    """
    將原始 RDI 轉成比較接近 mmWave Demo Visualizer 的顯示方式。

    原始 rdi_raw shape:
        (range_bins, doppler_bins)

    轉換後 rdi_plot shape:
        (doppler_bins, range_bins)

    也就是：
        x 軸 = range
        y 軸 = doppler
    """

    # 1. Doppler 軸做 fftshift
    # 原本 0 Doppler 可能在邊界，shift 後會移到中間
    rdi_shift = np.fft.fftshift(rdi_raw, axes=1)

    # 2. 轉置
    # 原始 shape 是 range x doppler
    # Demo Visualizer 比較像 doppler x range
    rdi_plot = rdi_shift.T

    # 3. log scale
    # 原始 uint16 強度差異可能很大，log 後比較像 Demo Visualizer
    rdi_plot_db = 20 * np.log10(rdi_plot.astype(np.float32) + 1.0)

    return rdi_plot_db


# ============================================================
# 6. 解析 .dat 檔
# ============================================================

with open(file_path, "rb") as f:
    data = f.read()

offset = 0
frame_count = 0

rdi_frames = []
target_found = False

print("開始解析 .dat 檔...")

while True:
    idx = data.find(MAGIC, offset)

    if idx == -1:
        print("已經找不到更多 frame")
        break

    # 確認 header 長度足夠
    if idx + 40 > len(data):
        print("剩餘資料不足以解析 header")
        break

    # 解析 40 bytes header
    # TI UART packet header:
    # magicWord        uint64
    # version          uint32
    # totalPacketLen   uint32
    # platform         uint32
    # frameNumber      uint32
    # timeCpuCycles    uint32
    # numDetectedObj   uint32
    # numTLVs          uint32
    # subFrameNumber   uint32
    header = struct.unpack_from("<QIIIIIIII", data, idx)

    magic_word = header[0]
    version = header[1]
    packet_len = header[2]
    platform = header[3]
    frame_number = header[4]
    time_cpu_cycles = header[5]
    num_detected_obj = header[6]
    num_tlvs = header[7]
    sub_frame_number = header[8]

    # 確認整個 packet 是否完整
    if idx + packet_len > len(data):
        print(f"Frame {frame_count} 資料不完整，停止解析")
        break

    tlv_offset = idx + 40
    found_rdi_in_this_frame = False

    for tlv_i in range(num_tlvs):
        # 確認 TLV header 長度足夠
        if tlv_offset + 8 > idx + packet_len:
            print(f"Frame {frame_count}: TLV header 不完整")
            break

        tlv_type, tlv_length = struct.unpack_from("<II", data, tlv_offset)
        tlv_offset += 8

        # 確認 TLV data 長度足夠
        if tlv_offset + tlv_length > idx + packet_len:
            print(f"Frame {frame_count}: TLV data 不完整")
            break

        tlv_data = data[tlv_offset: tlv_offset + tlv_length]
        tlv_offset += tlv_length

        # TLV type 5 通常是 Range-Doppler Heatmap
        if tlv_type == 5:
            rdi = np.frombuffer(tlv_data, dtype=np.uint16)

            expected_size = RANGE_BINS * DOPPLER_BINS

            if rdi.size != expected_size:
                print(f"Frame {frame_count}: RDI size 不符合")
                print("目前 size =", rdi.size)
                print("預期 size =", expected_size)
                print("可能 shape：")

                candidates = [
                    (64, 64),
                    (128, 32),
                    (256, 16),
                    (32, 128),
                    (16, 256),
                ]

                for rb, db in candidates:
                    if rb * db == rdi.size:
                        print(f"  RANGE_BINS={rb}, DOPPLER_BINS={db}")

                continue

            # 原始 RDI: range x doppler
            rdi = rdi.reshape((RANGE_BINS, DOPPLER_BINS))

            # 轉成顯示用版本
            rdi_plot = convert_rdi_to_visualizer_style(rdi)

            rdi_frames.append({
                "frame_index": frame_count,
                "frame_number": frame_number,
                "rdi_raw": rdi,
                "rdi_plot": rdi_plot,
                "packet_len": packet_len,
                "num_tlvs": num_tlvs,
            })

            found_rdi_in_this_frame = True

            # 印指定 frame 資訊
            if PRINT_TARGET_FRAME and frame_count == TARGET_FRAME and not target_found:
                target_found = True

                print("=" * 70)
                print(f"找到目標 Frame index：{TARGET_FRAME}")
                print(f"雷達封包中的 frame_number：{frame_number}")
                print(f"packet_len：{packet_len}")
                print(f"num_tlvs：{num_tlvs}")
                print(f"TLV type：{tlv_type}")
                print(f"TLV length：{tlv_length} bytes")
                print(f"RDI raw shape：{rdi.shape}")
                print(f"RDI plot shape：{rdi_plot.shape}")
                print(f"RDI raw min：{np.min(rdi)}")
                print(f"RDI raw max：{np.max(rdi)}")
                print(f"RDI raw mean：{np.mean(rdi):.2f}")
                print(f"RDI plot dB min：{np.min(rdi_plot):.2f}")
                print(f"RDI plot dB max：{np.max(rdi_plot):.2f}")
                print(f"RDI plot dB mean：{np.mean(rdi_plot):.2f}")
                print("=" * 70)

                print("\n原始 RDI 左上角 10x10 數值：")
                print(rdi[:10, :10])

                print("\n原始 RDI 最大值位置：")
                max_pos = np.unravel_index(np.argmax(rdi), rdi.shape)
                print("range_bin, doppler_bin =", max_pos)
                print("max value =", rdi[max_pos])

                # 轉成實際座標
                max_range_bin = max_pos[0]
                max_doppler_bin = max_pos[1]

                range_m = max_range_bin * RANGE_RESOLUTION_M

                # Doppler bin 經過 fftshift 後，中心是 0
                shifted_doppler_bin = max_doppler_bin - (DOPPLER_BINS // 2)
                doppler_mps = shifted_doppler_bin * DOPPLER_RESOLUTION_MPS

                print("\n最大值大約對應座標：")
                print(f"Range ≈ {range_m:.3f} m")
                print(f"Doppler ≈ {doppler_mps:.3f} m/s")

                if PRINT_FULL_MATRIX:
                    print("\n完整原始 RDI 矩陣：")
                    print(rdi)

    if not found_rdi_in_this_frame:
        print(f"Frame {frame_count}: 沒有找到 TLV type 5")

    frame_count += 1
    offset = idx + packet_len


print()
print(f"總共解析到 {frame_count} 個完整 packet")
print(f"總共取得 {len(rdi_frames)} 個 RDI frame")


# ============================================================
# 7. 沒有資料就結束
# ============================================================

if len(rdi_frames) == 0:
    print("沒有可播放的 RDI frame")
    exit()


# ============================================================
# 8. 設定顏色範圍
# ============================================================

if FIX_COLOR_SCALE:
    all_values = np.concatenate([
        item["rdi_plot"].ravel() for item in rdi_frames
    ])

    vmin = np.percentile(all_values, COLOR_PERCENTILE_LOW)
    vmax = np.percentile(all_values, COLOR_PERCENTILE_HIGH)

    print(f"固定顏色範圍：vmin={vmin:.2f}, vmax={vmax:.2f}")
else:
    vmin = None
    vmax = None


# ============================================================
# 9. 動態播放
# ============================================================

plt.ion()

fig, ax = plt.subplots(figsize=(8, 6))

first_item = rdi_frames[0]
first_rdi_plot = first_item["rdi_plot"]

im = ax.imshow(
    first_rdi_plot,
    aspect="auto",
    origin="lower",
    cmap="jet",
    extent=[
        0,
        MAX_RANGE_M,
        -MAX_DOPPLER_MPS,
        MAX_DOPPLER_MPS,
    ],
    vmin=vmin,
    vmax=vmax,
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Magnitude (dB)")

ax.set_xlabel("Range (meters)")
ax.set_ylabel("Doppler (m/s)")

title = ax.set_title(
    f"Range-Doppler Heatmap - Frame 0 "
    f"(frame_number={first_item['frame_number']})"
)

# 畫一條 Doppler = 0 的水平線
ax.axhline(0, linestyle="--", linewidth=1)

plt.tight_layout()

print()
print("開始播放 RDI...")
print("關閉圖視窗即可停止。")
print("Ctrl + C 也可以中止。")

try:
    while True:
        for i, item in enumerate(rdi_frames):
            rdi_plot = item["rdi_plot"]
            frame_number = item["frame_number"]

            im.set_data(rdi_plot)

            # 如果不固定顏色範圍，就每一幀自動調整
            if not FIX_COLOR_SCALE:
                current_vmin = np.percentile(rdi_plot, COLOR_PERCENTILE_LOW)
                current_vmax = np.percentile(rdi_plot, COLOR_PERCENTILE_HIGH)
                im.set_clim(current_vmin, current_vmax)

            title.set_text(
                f"Range-Doppler Heatmap - Frame {i} "
                f"(frame_number={frame_number})"
            )

            fig.canvas.draw()
            fig.canvas.flush_events()

            plt.pause(1.0 / FPS)

            # 如果視窗被關掉，就結束播放
            if not plt.fignum_exists(fig.number):
                raise SystemExit

        if not LOOP_PLAYBACK:
            break

except KeyboardInterrupt:
    print("播放已由 Ctrl + C 中止")

except SystemExit:
    print("播放視窗已關閉")

finally:
    plt.ioff()
    plt.show()