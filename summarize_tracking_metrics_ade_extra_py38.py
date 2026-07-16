# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Tuple


# =========================
# 1. 使用者設定
# =========================

# 你的 Record 資料夾
RECORD_ROOT = Path(r"C:\Users\mc2\Desktop\ti_tracking\Record")

# 要納入統計的受測者資料夾名稱
SUBJECTS = ["matt", "ben", "frank", "u", "jianhua"]

# 輸出資料夾名稱，會建立在 RECORD_ROOT 底下
OUTPUT_FOLDER_PREFIX = "tracking_metrics_summary"

# 軌跡長度相似度（Trajectory Length Similarity, TLS）會比較
# Prediction 與 Ground Truth 的累積軌跡長度是否接近。
# TLS 範圍為 0%~100%，數值越高代表預測軌跡總移動量越接近真實軌跡。
# 軌跡顯示順序與名稱
TRAJECTORY_ORDER = [
    ("circle", "圓形"),
    ("inf", "無限符號"),
    ("hori", "左右往返"),
    ("verti", "前後推進"),
    ("random", "隨機移動"),
]


# =========================
# 2. 工具函式
# =========================

def detect_trajectory_name(text: str) -> Tuple[str, str]:
    """
    從資料夾名稱或 csv 裡的 trajectory 欄位判斷軌跡類型。
    回傳：(trajectory_key, trajectory_label)
    """
    t = str(text).lower()

    aliases = {
        "circle": ["circle", "圓"],
        "inf": ["inf", "infinity", "eight", "無限"],
        "hori": ["hori", "horizontal", "left", "right", "左右"],
        "verti": ["verti", "vertical", "front", "back", "前後", "上下"],
        "random": ["random", "rand", "隨機"],
    }

    label_map = dict(TRAJECTORY_ORDER)

    for key, words in aliases.items():
        if any(w in t for w in words):
            return key, label_map[key]

    return "unknown", "未知軌跡"


def read_prediction_csv(csv_path: Path, subject: str) -> pd.DataFrame:
    """
    讀取單一 offline_eval_predictions.csv，並補齊必要欄位。
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 判斷軌跡類型：優先看 csv 的 trajectory 欄位，否則看資料夾名稱
    if "trajectory" in df.columns and len(df) > 0:
        traj_source = str(df["trajectory"].iloc[0])
    else:
        traj_source = csv_path.parent.name

    traj_key, traj_label = detect_trajectory_name(traj_source)

    df["subject"] = subject
    df["trajectory_key"] = traj_key
    df["trajectory_label"] = traj_label
    df["source_file"] = str(csv_path)

    # 若沒有誤差欄位，則由 GT 與 Prediction 欄位重新計算
    if "angle_abs_error_deg" not in df.columns:
        df["angle_abs_error_deg"] = (df["pred_angle_deg"] - df["gt_angle_deg"]).abs()

    if "distance_abs_error_cm" not in df.columns:
        df["distance_abs_error_cm"] = (df["pred_distance_cm"] - df["gt_distance_cm"]).abs()

    required_xy = {"pred_x_cm", "gt_x_cm", "pred_y_cm", "gt_y_cm"}
    if not required_xy.issubset(df.columns):
        raise ValueError(
            f"{csv_path} 缺少 pred_x_cm / gt_x_cm / pred_y_cm / gt_y_cm，"
            "無法計算 X、Y、平面定位 ADE、離散弗雷歇距離與 TLS。"
        )

    if "x_abs_error_cm" not in df.columns:
        df["x_abs_error_cm"] = (df["pred_x_cm"] - df["gt_x_cm"]).abs()

    if "y_abs_error_cm" not in df.columns:
        df["y_abs_error_cm"] = (df["pred_y_cm"] - df["gt_y_cm"]).abs()

    if "xy_error_cm" not in df.columns:
        dx = df["pred_x_cm"] - df["gt_x_cm"]
        dy = df["pred_y_cm"] - df["gt_y_cm"]
        df["xy_error_cm"] = np.sqrt(dx ** 2 + dy ** 2)

    # 保留原始順序，避免後續計算軌跡型指標時跨段或排序錯亂
    df["row_order_in_file"] = np.arange(len(df))

    return df


def _sort_one_track(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算軌跡型指標前的排序規則。
    優先用 frame_index，其次 time_s，若都沒有則用檔案內原始順序。
    """
    if "frame_index" in df.columns:
        return df.sort_values("frame_index")
    if "time_s" in df.columns:
        return df.sort_values("time_s")
    return df.sort_values("row_order_in_file")


def _discrete_frechet_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    計算離散弗雷歇距離（Discrete Fréchet Distance）。

    概念可理解為在保持軌跡點順序的前提下，
    兩條曲線同步前進時所需面對的最大距離。
    單位為 cm，數值越小代表兩條軌跡的幾何形狀越接近；
    但它會受到局部最大誤差影響，因此可用來觀察最差偏離程度。
    """
    n, m = len(pred), len(gt)
    if n == 0 or m == 0:
        return 0.0

    dist = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    ca = np.full((n, m), np.inf, dtype=float)
    ca[0, 0] = dist[0, 0]

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue

            candidates = []
            if i > 0:
                candidates.append(ca[i - 1, j])
            if j > 0:
                candidates.append(ca[i, j - 1])
            if i > 0 and j > 0:
                candidates.append(ca[i - 1, j - 1])

            ca[i, j] = max(dist[i, j], min(candidates))

    return float(ca[n - 1, m - 1])


def trajectory_frechet_distance(group: pd.DataFrame) -> float:
    """
    計算平均離散弗雷歇距離（cm）。

    對每個 source_file 分別計算離散弗雷歇距離，再取平均。
    這樣在受測者平均、軌跡平均與總平均時，不會讓單一檔案的極端最大誤差完全主導結果。
    """
    values = []

    if "source_file" in group.columns:
        groups = group.groupby("source_file", sort=False)
    else:
        groups = [("single", group)]

    for _, one_track in groups:
        one_track = _sort_one_track(one_track)
        if len(one_track) < 2:
            continue

        pred = one_track[["pred_x_cm", "pred_y_cm"]].to_numpy(dtype=float)
        gt = one_track[["gt_x_cm", "gt_y_cm"]].to_numpy(dtype=float)

        values.append(_discrete_frechet_distance(pred, gt))

    if not values:
        return 0.0
    return float(np.mean(values))


def trajectory_length_similarity_percent(group: pd.DataFrame) -> float:
    """
    計算軌跡長度相似度（Trajectory Length Similarity, TLS），單位為 %。

    TLS 用於比較 Prediction 與 Ground Truth 的累積移動距離是否接近。
    先分別計算每一條軌跡的累積路徑長度：

        L_pred = sum(||pred_i - pred_{i-1}||)
        L_gt   = sum(||gt_i - gt_{i-1}||)

    再以兩者差異相對於總長度進行正規化：

        TLS = (1 - sum(|L_pred - L_gt|) / sum(L_pred + L_gt)) * 100%

    TLS 範圍為 0%~100%，數值越高表示預測軌跡的總移動量越接近 Ground Truth。
    若 TLS 較低，可能表示預測軌跡移動量不足、過度平滑，或因抖動造成累積移動量過大。

    當 group 合併多個 csv 時，會分 source_file 分段計算，避免跨檔案相減。
    """
    numerator = 0.0
    denominator = 0.0

    if "source_file" in group.columns:
        groups = group.groupby("source_file", sort=False)
    else:
        groups = [("single", group)]

    for _, one_track in groups:
        one_track = _sort_one_track(one_track)
        if len(one_track) < 2:
            continue

        pred = one_track[["pred_x_cm", "pred_y_cm"]].to_numpy(dtype=float)
        gt = one_track[["gt_x_cm", "gt_y_cm"]].to_numpy(dtype=float)

        pred_step = np.diff(pred, axis=0)
        gt_step = np.diff(gt, axis=0)

        pred_length = float(np.linalg.norm(pred_step, axis=1).sum())
        gt_length = float(np.linalg.norm(gt_step, axis=1).sum())

        numerator += abs(pred_length - gt_length)
        denominator += (pred_length + gt_length)

    if denominator == 0.0:
        # Prediction 與 Ground Truth 都沒有移動時，視為軌跡長度完全一致。
        return 100.0

    tls = (1.0 - np.clip(numerator / denominator, 0.0, 1.0)) * 100.0
    return float(tls)

def summarize_frame_group(group: pd.DataFrame) -> dict:
    """
    用逐 frame 資料計算一組資料的統計結果。

    輸出項目：
    角度 MAE（SD）、距離 MAE（SD）、X MAE（SD）、Y MAE（SD）、
    平面定位 ADE（SD）、離散弗雷歇距離、TLS。
    """
    return {
        "frames": int(len(group)),

        "angle_mae": float(group["angle_abs_error_deg"].mean()),
        "angle_sd": float(group["angle_abs_error_deg"].std(ddof=0)),

        "distance_mae": float(group["distance_abs_error_cm"].mean()),
        "distance_sd": float(group["distance_abs_error_cm"].std(ddof=0)),

        "x_mae": float(group["x_abs_error_cm"].mean()),
        "x_sd": float(group["x_abs_error_cm"].std(ddof=0)),

        "y_mae": float(group["y_abs_error_cm"].mean()),
        "y_sd": float(group["y_abs_error_cm"].std(ddof=0)),

        # xy_error_cm 為每一幀 Prediction 與 Ground Truth 在 X-Y 平面的歐氏距離，
        # 對其取平均即為平面定位 ADE（Average Displacement Error）。
        "xy_ade": float(group["xy_error_cm"].mean()),
        "xy_ade_sd": float(group["xy_error_cm"].std(ddof=0)),

        "frechet_cm": trajectory_frechet_distance(group),
        "tls_percent": trajectory_length_similarity_percent(group),
    }


def add_formatted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    加入可直接貼到論文或簡報表格的字串欄位。
    """
    out = df.copy()

    out["角度 MAE ± SD (deg)"] = out.apply(
        lambda r: f"{r['angle_mae']:.2f}° ± {r['angle_sd']:.2f}°", axis=1
    )
    out["距離 MAE ± SD (cm)"] = out.apply(
        lambda r: f"{r['distance_mae']:.2f} ± {r['distance_sd']:.2f}", axis=1
    )
    out["X MAE ± SD (cm)"] = out.apply(
        lambda r: f"{r['x_mae']:.2f} ± {r['x_sd']:.2f}", axis=1
    )
    out["Y MAE ± SD (cm)"] = out.apply(
        lambda r: f"{r['y_mae']:.2f} ± {r['y_sd']:.2f}", axis=1
    )
    out["平面定位 ADE ± SD (cm)"] = out.apply(
        lambda r: f"{r['xy_ade']:.2f} ± {r['xy_ade_sd']:.2f}", axis=1
    )
    out["離散弗雷歇距離 (cm)"] = out["frechet_cm"].map(lambda x: f"{x:.2f}")
    out["軌跡長度相似度 TLS (%)"] = out["tls_percent"].map(lambda x: f"{x:.2f}%")

    return out


def export_pretty_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """
    用 utf-8-sig 輸出，讓 Excel 開啟中文不亂碼。
    """
    df.to_csv(path, index=index, encoding="utf-8-sig")


# =========================
# 3. 主流程
# =========================

def main() -> None:
    if not RECORD_ROOT.exists():
        raise FileNotFoundError(f"找不到 RECORD_ROOT：{RECORD_ROOT}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RECORD_ROOT / f"{OUTPUT_FOLDER_PREFIX}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []

    # 讀取五位受測者底下所有 offline_eval_predictions.csv
    for subject in SUBJECTS:
        subject_dir = RECORD_ROOT / subject

        if not subject_dir.exists():
            print(f"[警告] 找不到受測者資料夾，略過：{subject_dir}")
            continue

        csv_files = sorted(subject_dir.rglob("offline_eval_predictions.csv"))

        if not csv_files:
            print(f"[警告] {subject} 找不到 offline_eval_predictions.csv")
            continue

        for csv_path in csv_files:
            try:
                df = read_prediction_csv(csv_path, subject)
                all_frames.append(df)
                print(f"[讀取] {subject} / {csv_path.parent.name} / {len(df)} frames")
            except Exception as e:
                print(f"[錯誤] 無法讀取 {csv_path}: {e}")

    if not all_frames:
        raise RuntimeError("沒有讀到任何 offline_eval_predictions.csv，請確認資料夾路徑與檔名。")

    all_df = pd.concat(all_frames, ignore_index=True)

    # 排除未知軌跡，避免資料夾名稱不符合時污染統計
    unknown = all_df[all_df["trajectory_key"] == "unknown"]
    if len(unknown) > 0:
        unknown_path = output_dir / "unknown_trajectory_rows.csv"
        export_pretty_csv(unknown, unknown_path)
        print(f"[警告] 有 {len(unknown)} 筆資料無法判斷軌跡類型，已另存：{unknown_path}")

    all_df = all_df[all_df["trajectory_key"] != "unknown"].copy()

    # 設定排序用欄位
    trajectory_rank = {key: i for i, (key, _) in enumerate(TRAJECTORY_ORDER)}
    subject_rank = {s: i for i, s in enumerate(SUBJECTS)}
    all_df["trajectory_rank"] = all_df["trajectory_key"].map(trajectory_rank)
    all_df["subject_rank"] = all_df["subject"].map(subject_rank)

    # =========================
    # A. 每位受測者 × 每種軌跡
    # =========================
    rows = []
    grouped = all_df.groupby(["subject", "subject_rank", "trajectory_key", "trajectory_rank", "trajectory_label"], dropna=False)

    for (subject, s_rank, t_key, t_rank, t_label), g in grouped:
        row = {
            "subject": subject,
            "subject_rank": int(s_rank),
            "trajectory_key": t_key,
            "trajectory_rank": int(t_rank),
            "trajectory_label": t_label,
        }
        row.update(summarize_frame_group(g))
        rows.append(row)

    subject_trajectory = pd.DataFrame(rows).sort_values(["subject_rank", "trajectory_rank"]).reset_index(drop=True)
    subject_trajectory_fmt = add_formatted_columns(subject_trajectory)

    # =========================
    # B. 每位受測者平均：合併該受測者五種軌跡所有 frames
    # =========================
    rows = []
    for (subject, s_rank), g in all_df.groupby(["subject", "subject_rank"], dropna=False):
        row = {
            "subject": subject,
            "subject_rank": int(s_rank),
        }
        row.update(summarize_frame_group(g))
        rows.append(row)

    subject_average = pd.DataFrame(rows).sort_values("subject_rank").reset_index(drop=True)
    subject_average_fmt = add_formatted_columns(subject_average)

    # 在受測者平均表最後加入總平均：合併 25 個軌跡所有 frames
    total_row = {"subject": "平均", "subject_rank": 999}
    total_row.update(summarize_frame_group(all_df))
    subject_average_with_total = pd.concat(
        [subject_average, pd.DataFrame([total_row])],
        ignore_index=True
    )
    subject_average_with_total_fmt = add_formatted_columns(subject_average_with_total)

    # =========================
    # C. 每種軌跡五人平均：合併五位受測者同軌跡所有 frames
    # =========================
    rows = []
    for (t_key, t_rank, t_label), g in all_df.groupby(["trajectory_key", "trajectory_rank", "trajectory_label"], dropna=False):
        row = {
            "trajectory_key": t_key,
            "trajectory_rank": int(t_rank),
            "trajectory_label": t_label,
        }
        row.update(summarize_frame_group(g))
        rows.append(row)

    trajectory_average = pd.DataFrame(rows).sort_values("trajectory_rank").reset_index(drop=True)
    trajectory_average_fmt = add_formatted_columns(trajectory_average)

    # 在軌跡平均表最後加入總平均：合併 25 個軌跡所有 frames
    total_traj_row = {
        "trajectory_key": "total",
        "trajectory_rank": 999,
        "trajectory_label": "平均",
    }
    total_traj_row.update(summarize_frame_group(all_df))
    trajectory_average_with_total = pd.concat(
        [trajectory_average, pd.DataFrame([total_traj_row])],
        ignore_index=True
    )
    trajectory_average_with_total_fmt = add_formatted_columns(trajectory_average_with_total)

    # =========================
    # D. 25 個軌跡總平均
    # =========================
    total_average = pd.DataFrame([{
        "name": "25 個軌跡總平均",
        **summarize_frame_group(all_df)
    }])
    total_average_fmt = add_formatted_columns(total_average)

    # =========================
    # 4. 輸出檔案
    # =========================

    # 原始逐 frame 合併資料，方便追蹤
    export_pretty_csv(all_df, output_dir / "00_all_frames_combined.csv")

    # 數值版 CSV
    export_pretty_csv(subject_trajectory, output_dir / "01_subject_trajectory_metrics_numeric.csv")
    export_pretty_csv(subject_average_with_total, output_dir / "02_subject_average_numeric.csv")
    export_pretty_csv(trajectory_average_with_total, output_dir / "03_trajectory_average_numeric.csv")
    export_pretty_csv(total_average, output_dir / "04_total_average_numeric.csv")

    # 論文表格版 CSV
    metric_cols = [
        "角度 MAE ± SD (deg)",
        "距離 MAE ± SD (cm)",
        "X MAE ± SD (cm)",
        "Y MAE ± SD (cm)",
        "平面定位 ADE ± SD (cm)",
        "離散弗雷歇距離 (cm)",
        "軌跡長度相似度 TLS (%)",
        "frames",
    ]

    subject_trajectory_cols = ["subject", "trajectory_label"] + metric_cols
    subject_average_cols = ["subject"] + metric_cols
    trajectory_average_cols = ["trajectory_label"] + metric_cols
    total_average_cols = ["name"] + metric_cols

    export_pretty_csv(
        subject_trajectory_fmt[subject_trajectory_cols],
        output_dir / "01_subject_trajectory_metrics_for_paper.csv"
    )
    export_pretty_csv(
        subject_average_with_total_fmt[subject_average_cols],
        output_dir / "02_subject_average_for_paper.csv"
    )
    export_pretty_csv(
        trajectory_average_with_total_fmt[trajectory_average_cols],
        output_dir / "03_trajectory_average_for_paper.csv"
    )
    export_pretty_csv(
        total_average_fmt[total_average_cols],
        output_dir / "04_total_average_for_paper.csv"
    )

    # Excel 多工作表
    excel_path = output_dir / "tracking_metrics_summary.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            subject_trajectory_fmt[subject_trajectory_cols].to_excel(writer, sheet_name="受測者x軌跡", index=False)
            subject_average_with_total_fmt[subject_average_cols].to_excel(writer, sheet_name="受測者平均", index=False)
            trajectory_average_with_total_fmt[trajectory_average_cols].to_excel(writer, sheet_name="軌跡平均", index=False)
            total_average_fmt[total_average_cols].to_excel(writer, sheet_name="總平均", index=False)

            subject_trajectory.to_excel(writer, sheet_name="numeric_受測者x軌跡", index=False)
            subject_average_with_total.to_excel(writer, sheet_name="numeric_受測者平均", index=False)
            trajectory_average_with_total.to_excel(writer, sheet_name="numeric_軌跡平均", index=False)
            total_average.to_excel(writer, sheet_name="numeric_總平均", index=False)

        print(f"[輸出] Excel：{excel_path}")
    except Exception as e:
        print(f"[警告] Excel 輸出失敗，CSV 仍已輸出。原因：{e}")

    # TXT 報告
    report_path = output_dir / "tracking_metrics_report.txt"
    with open(report_path, "w", encoding="utf-8-sig") as f:
        f.write("毫米波雷達手部追蹤測試結果彙整\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"資料來源：{RECORD_ROOT}\n")
        f.write(f"受測者：{', '.join(SUBJECTS)}\n")
        f.write("表格欄位：角度 MAE（SD）、距離 MAE（SD）、X MAE（SD）、Y MAE（SD）、平面定位 ADE（SD）、離散弗雷歇距離、軌跡長度相似度 TLS。\n")
        f.write("統計方式：由逐 frame 誤差重新計算；ADE 為平面座標歐氏距離平均；離散弗雷歇距離與 TLS 皆以各 csv 檔案分段計算，避免跨檔案相減。\n\n")

        f.write("一、各軌跡五人平均\n")
        f.write("-" * 60 + "\n")
        f.write(trajectory_average_with_total_fmt[trajectory_average_cols].to_string(index=False))
        f.write("\n\n")

        f.write("二、各受測者平均\n")
        f.write("-" * 60 + "\n")
        f.write(subject_average_with_total_fmt[subject_average_cols].to_string(index=False))
        f.write("\n\n")

        f.write("三、25 個軌跡總平均\n")
        f.write("-" * 60 + "\n")
        f.write(total_average_fmt[total_average_cols].to_string(index=False))
        f.write("\n\n")

        f.write("四、各受測者 × 各軌跡完整結果\n")
        f.write("-" * 60 + "\n")
        f.write(subject_trajectory_fmt[subject_trajectory_cols].to_string(index=False))
        f.write("\n")

    print(f"[輸出] 報告：{report_path}")
    print(f"\n完成。所有結果已輸出到：\n{output_dir}")


if __name__ == "__main__":
    main()
