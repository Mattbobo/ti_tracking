import numpy as np, h5py, csv, matplotlib.pyplot as plt, pickle, zmq
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import math

# === ZeroMQ Client (連線到模型伺服器) ===
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://localhost:5555")

def ask_angle(seq):
    sock.send(pickle.dumps(('angle', seq), protocol=4))
    angle, _ = sock.recv_pyobj()
    return float(angle)

def ask_dist(seq):
    sock.send(pickle.dumps(('dist', seq), protocol=4))
    _, dist = sock.recv_pyobj()
    return float(dist)

# === Normalization (與訓練一致) ===
mean_arr = np.array([62.93287659, 60.18955612], dtype=np.float32)
std_arr  = np.array([10.94364929, 10.48015118], dtype=np.float32)
SEQ_LEN = 20
FPS = 20.0   # 取樣頻率

# === 繪圖時間軸設定 ===
# 若你確定這段測試資料代表完整 10 秒，保持 True，會把有效預測段等比例顯示成 0~10 秒。
# 若你想嚴格照 FPS 顯示真實有效預測時間，改成 False。
PLOT_DURATION = 10.0
RESCALE_TIME_TO_10S = True

# === 評估輸出設定 ===
# 可手動填入軌跡名稱，例如：圓形、無限符號、左右往返、前後推進、隨機移動。
# 若保持空字串，程式會自動使用資料夾名稱作為軌跡名稱。
TRAJECTORY_NAME = ""
OVERLAP_THRESHOLD_CM = 2.0  # 軌跡重合度門檻：平面誤差 <= 2 cm 視為重合


# === 輸入檔案 ===
h5_path = r"C:\Users\mc2\Desktop\ti_tracking\Record\jianhua\jianhua_random\data_20260610_132644.h5"
csv_path = r"C:\Users\mc2\Desktop\ti_tracking\Record\jianhua\jianhua_random\records.csv"

h5_dir = Path(h5_path).resolve().parent
csv_dir = Path(csv_path).resolve().parent
if h5_dir != csv_dir:
    raise ValueError(f"h5 and csv are in different folders: {h5_dir} != {csv_dir}")
output_dir = h5_dir
trajectory_name = TRAJECTORY_NAME.strip() if TRAJECTORY_NAME.strip() else h5_dir.name

def safe_std(values):
    """計算樣本標準差；若資料數不足，回傳 0。"""
    values = np.asarray(values, dtype=np.float64)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

# === 讀取資料 ===
with h5py.File(h5_path, 'r') as f:
    ds = np.array(f['DS1'])
N = ds.shape[0]
angles, dists = [], []

with open(csv_path, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)

    for row in reader:
        angles.append(float(row['angle_deg']))
        dists.append(float(row['distance_cm']))

angles = np.array(angles, dtype=np.float32)
dists  = np.array(dists, dtype=np.float32)

# === 預測 ===
pred_a, pred_d = [], []
for i in range(SEQ_LEN - 1, N):
    seq = ds[i - SEQ_LEN + 1 : i + 1].astype(np.float32)
    seq_norm = (seq - mean_arr[None, :, None, None]) / std_arr[None, :, None, None]
    pred_a.append(ask_angle(seq_norm))
    pred_d.append(ask_dist(seq_norm[:, 0:1, :, :]))
pred_a, pred_d = np.array(pred_a), np.array(pred_d)

# === 對齊 Label ===
gt_a = angles[SEQ_LEN - 1 :]
gt_d = dists[SEQ_LEN - 1 :]
# === 建立時間軸 ===
# 原本 frames / FPS 會因為 SEQ_LEN 少掉前 19 個 frame，導致最後不到 10 秒。
# 用 linspace 可讓論文/簡報圖完整顯示 0~10 秒。
if RESCALE_TIME_TO_10S:
    time_s = np.linspace(0.0, PLOT_DURATION, len(pred_a), endpoint=True)
else:
    frames = np.arange(len(pred_a))
    time_s = frames / FPS

def format_time_axis(ax):
    # 去掉 Matplotlib 預設左右留白，並固定顯示 0~10 秒與 10 秒刻度。
    ax.set_xlim(0.0, PLOT_DURATION)
    ax.set_xticks(np.arange(0.0, PLOT_DURATION + 0.001, 2.0))
    ax.margins(x=0)

def polar_to_xy_cm(angle_deg, dist_cm):
    rad = np.deg2rad(angle_deg)
    x = dist_cm * np.sin(rad)
    y = dist_cm * np.cos(rad)
    return x, y

gt_x, gt_y = polar_to_xy_cm(gt_a, gt_d)
pred_x, pred_y = polar_to_xy_cm(pred_a, pred_d)

# === 指標 ===
angle_err_abs = np.abs(gt_a - pred_a)
dist_err_abs = np.abs(gt_d - pred_d)
x_err_abs = np.abs(gt_x - pred_x)
y_err_abs = np.abs(gt_y - pred_y)
xy_err_cm = np.sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)

mae_a = float(np.mean(angle_err_abs))
mae_d = float(np.mean(dist_err_abs))
mae_xy_cm = float(np.mean(xy_err_cm))
mae_x = float(np.mean(x_err_abs))
mae_y = float(np.mean(y_err_abs))

sd_a = safe_std(angle_err_abs)
sd_d = safe_std(dist_err_abs)
sd_xy_cm = safe_std(xy_err_cm)
sd_x = safe_std(x_err_abs)
sd_y = safe_std(y_err_abs)

p95_xy_cm = float(np.percentile(xy_err_cm, 95))
overlap_ratio_2cm = float(np.mean(xy_err_cm <= OVERLAP_THRESHOLD_CM) * 100.0)

rmse_a = math.sqrt(mean_squared_error(gt_a, pred_a))
rmse_d = math.sqrt(mean_squared_error(gt_d, pred_d))
r2_a = r2_score(gt_a, pred_a)
r2_d = r2_score(gt_d, pred_d)

print(f"Trajectory: {trajectory_name}")
print(f"XY: MAE={mae_xy_cm:.2f}±{sd_xy_cm:.2f} cm, P95={p95_xy_cm:.2f} cm, <= {OVERLAP_THRESHOLD_CM:.1f} cm overlap={overlap_ratio_2cm:.2f}%")
print(f"X: MAE={mae_x:.2f}±{sd_x:.2f} cm")
print(f"Y: MAE={mae_y:.2f}±{sd_y:.2f} cm")
print(f"Angle: MAE={mae_a:.2f}±{sd_a:.2f}°, RMSE={rmse_a:.2f}, R²={r2_a:.3f}")
print(f"Range: MAE={mae_d:.2f}±{sd_d:.2f} cm, RMSE={rmse_d:.2f}, R²={r2_d:.3f}")

# === 儲存逐幀模型預測結果 ===
predictions_path = output_dir / 'offline_eval_predictions.csv'
frame_indices = np.arange(SEQ_LEN - 1, SEQ_LEN - 1 + len(pred_a))

with open(predictions_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow([
        'trajectory', 'frame_index', 'time_s',
        'gt_angle_deg', 'pred_angle_deg', 'angle_abs_error_deg',
        'gt_distance_cm', 'pred_distance_cm', 'distance_abs_error_cm',
        'gt_x_cm', 'pred_x_cm', 'x_abs_error_cm',
        'gt_y_cm', 'pred_y_cm', 'y_abs_error_cm',
        'xy_error_cm', f'within_{OVERLAP_THRESHOLD_CM:g}cm'
    ])

    for row in zip(
        frame_indices, time_s,
        gt_a, pred_a, angle_err_abs,
        gt_d, pred_d, dist_err_abs,
        gt_x, pred_x, x_err_abs,
        gt_y, pred_y, y_err_abs,
        xy_err_cm, xy_err_cm <= OVERLAP_THRESHOLD_CM
    ):
        (frame_idx, t,
         ga, pa, ea,
         gd, pd, ed,
         gx, px, ex,
         gy, py, ey,
         exy, within) = row

        writer.writerow([
            trajectory_name, int(frame_idx), f'{float(t):.6f}',
            f'{float(ga):.6f}', f'{float(pa):.6f}', f'{float(ea):.6f}',
            f'{float(gd):.6f}', f'{float(pd):.6f}', f'{float(ed):.6f}',
            f'{float(gx):.6f}', f'{float(px):.6f}', f'{float(ex):.6f}',
            f'{float(gy):.6f}', f'{float(py):.6f}', f'{float(ey):.6f}',
            f'{float(exy):.6f}', int(bool(within))
        ])

print(f"Saved predictions: {predictions_path}")

# === 儲存單一軌跡表現摘要，可直接貼到表 4-1 ===
summary_path = output_dir / 'offline_eval_metrics_summary.txt'
summary_header = (
    '軌跡類型\t'
    '角度 MAE ± SD (deg)\t'
    '距離 MAE ± SD (cm)\t'
    '平面定位 MAE ± SD (cm)\t'
    'P95 平面誤差 (cm)\t'
    f'{OVERLAP_THRESHOLD_CM:g} cm 內軌跡重合度 (%)'
)
summary_row = (
    f'{trajectory_name}\t'
    f'{mae_a:.2f} ± {sd_a:.2f}\t'
    f'{mae_d:.2f} ± {sd_d:.2f}\t'
    f'{mae_xy_cm:.2f} ± {sd_xy_cm:.2f}\t'
    f'{p95_xy_cm:.2f}\t'
    f'{overlap_ratio_2cm:.2f}'
)

with open(summary_path, 'w', encoding='utf-8-sig') as f:
    f.write('表 4-1 單一軌跡評估結果\n')
    f.write('說明：標準差以逐幀絕對誤差計算；平面定位誤差為 sqrt((x_pred-x_gt)^2+(y_pred-y_gt)^2)。\n')
    f.write(f'軌跡重合度定義：平面定位誤差 <= {OVERLAP_THRESHOLD_CM:g} cm 的 frame 比例。\n\n')
    f.write(summary_header + '\n')
    f.write(summary_row + '\n')

print(f"Saved summary: {summary_path}")
print(summary_header)
print(summary_row)

# === 繪圖 ===
plt.figure(figsize=(10,7.5))
plt.suptitle('Radar Model vs Ground Truth', fontsize=15, fontweight='bold')

# --- 上圖：Angle ---
ax1 = plt.subplot(2,1,1)
ax1.plot(time_s, gt_a, label='Ground Truth (Angle)', linewidth=1.8)
ax1.plot(time_s, pred_a, '--', label='Prediction (Angle)', linewidth=1.8)
ax1.set_ylabel('Angle (°)')
ax1.set_xlabel('Time (s)')
ax1.grid(True)
ax1.legend(loc='upper right', fontsize=10, frameon=True)
ax1.text(0.98, 0.05, f'MAE = {mae_a:.2f}°',
         transform=ax1.transAxes, ha='right', va='bottom', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# ✅ 固定角度刻度與範圍
ax1.set_ylim(-45, 45)
ax1.set_yticks(np.arange(-45, 46, 15))
format_time_axis(ax1)

# --- 下圖：Range ---
ax2 = plt.subplot(2,1,2)
ax2.plot(time_s, gt_d, label='Ground Truth (Range)', linewidth=1.8)
ax2.plot(time_s, pred_d, '--', label='Prediction (Range)', linewidth=1.8)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Range (cm)')
ax2.grid(True)
ax2.legend(loc='upper right', fontsize=10, frameon=True)
ax2.text(0.98, 0.05, f'MAE = {mae_d:.2f} cm',
         transform=ax2.transAxes, ha='right', va='bottom', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# ✅ 固定距離刻度與範圍
ax2.set_ylim(0, 40)
ax2.set_yticks(np.arange(0, 41, 5))
format_time_axis(ax2)

plt.tight_layout(rect=[0,0,1,0.95])
result_path = output_dir / 'offline_eval_result.png'
plt.savefig(result_path, dpi=300, bbox_inches='tight')
print(f"Saved: {result_path}")

# === XY coordinate plot ===
plt.figure(figsize=(10,7.5))
plt.suptitle('XY Coordinates: Radar Model vs Ground Truth', fontsize=15, fontweight='bold')

ax3 = plt.subplot(2,1,1)
ax3.plot(time_s, gt_x, label='Ground Truth (X)', linewidth=1.8)
ax3.plot(time_s, pred_x, '--', label='Prediction (X)', linewidth=1.8)
ax3.set_ylabel('X (cm)')
ax3.set_xlabel('Time (s)')
ax3.grid(True)
ax3.legend(loc='upper right', fontsize=10, frameon=True)
ax3.text(0.98, 0.05, f'MAE = {mae_x:.2f} cm',
         transform=ax3.transAxes, ha='right', va='bottom', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
ax3.set_ylim(-30, 30)
ax3.set_yticks(np.arange(-30, 31, 10))
format_time_axis(ax3)

ax4 = plt.subplot(2,1,2)
ax4.plot(time_s, gt_y, label='Ground Truth (Y)', linewidth=1.8)
ax4.plot(time_s, pred_y, '--', label='Prediction (Y)', linewidth=1.8)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Y (cm)')
ax4.grid(True)
ax4.legend(loc='upper right', fontsize=10, frameon=True)
ax4.text(0.98, 0.05, f'MAE = {mae_y:.2f} cm',
         transform=ax4.transAxes, ha='right', va='bottom', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
ax4.set_ylim(0, 40)
ax4.set_yticks(np.arange(0, 41, 5))
format_time_axis(ax4)

plt.tight_layout(rect=[0,0,1,0.95])
xy_result_path = output_dir / 'offline_eval_xy_result.png'
plt.savefig(xy_result_path, dpi=300, bbox_inches='tight')
print(f"Saved: {xy_result_path}")
plt.show()
