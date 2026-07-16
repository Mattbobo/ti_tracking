# -*- coding: utf-8 -*-
"""
Preprocess distance data (channel 0) into NPZ splits + mean/std text.

Each raw frame sequence is split into contiguous train/val/test segments first,
then sliding windows are generated inside each split. This avoids putting
highly-overlapping neighboring windows into different train/val/test sets.

Output files:
- Process_data/Distance_data/train.npz  (X: [N,T,1,H,W], y: [N])
- Process_data/Distance_data/val.npz
- Process_data/Distance_data/test.npz
- Process_data/Distance_data/mean_std.txt  (two lines: train mean, train std)
"""
import csv
import glob
import os

import h5py
import numpy as np


# ===== Config =====
DATA_ROOT = 'Record'
OUT_DIR = os.path.join('Process_data', 'Distance_data')
SEQ_LEN = 20
STRIDE = 1
VAL_RATIO = 0.10
TEST_RATIO = 0.10
ROUND_LABEL_1_DECIMAL = True

os.makedirs(OUT_DIR, exist_ok=True)


def read_one_record_dir(rec_dir):
    """Return xs0:(frames,1,H,W), ds:(frames,) for one record folder."""
    h5_list = glob.glob(os.path.join(rec_dir, '*.h5'))
    csv_list = glob.glob(os.path.join(rec_dir, '*.csv'))
    if not h5_list or not csv_list:
        return None, None

    with h5py.File(h5_list[0], 'r') as f:
        arr = f['DS1'][:].astype(np.float32)  # (frames, C, H, W)
    xs0 = arr[:, 0:1, :, :]                  # (frames, 1, H, W)

    with open(csv_list[0], 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            idx = header.index('distance_cm')
        except ValueError:
            raise RuntimeError(f"'distance_cm' not found in {csv_list[0]}. Header: {header}")

        ds = np.asarray([float(row[idx]) for row in reader], dtype=np.float32)

    length = min(xs0.shape[0], ds.shape[0])
    xs0, ds = xs0[:length], ds[:length]

    if ROUND_LABEL_1_DECIMAL:
        ds = np.round(ds, 1)

    return xs0, ds


def split_sequence_lists(xs_list, ds_list):
    """Split each full record into train/val/test frame segments before windowing."""
    tr_xs, tr_ds = [], []
    va_xs, va_ds = [], []
    te_xs, te_ds = [], []

    for xs, ds in zip(xs_list, ds_list):
        n = xs.shape[0]
        n_val = max(int(VAL_RATIO * n), SEQ_LEN) if VAL_RATIO > 0 else 0
        n_test = max(int(TEST_RATIO * n), SEQ_LEN) if TEST_RATIO > 0 else 0
        n_train = n - n_val - n_test

        if n_train < SEQ_LEN:
            continue

        train_end = n_train
        val_end = train_end + n_val

        tr_xs.append(xs[:train_end])
        tr_ds.append(ds[:train_end])

        if n_val >= SEQ_LEN:
            va_xs.append(xs[train_end:val_end])
            va_ds.append(ds[train_end:val_end])

        if n_test >= SEQ_LEN:
            te_xs.append(xs[val_end:])
            te_ds.append(ds[val_end:])

    if not tr_xs or (VAL_RATIO > 0 and not va_xs) or (TEST_RATIO > 0 and not te_xs):
        raise RuntimeError("No usable train/val/test segments. Check SEQ_LEN and split ratios.")

    return (tr_xs, tr_ds), (va_xs, va_ds), (te_xs, te_ds)


def build_windows(xs_list, ds_list, seq_len, stride):
    """Build X windows and use the last frame distance as y."""
    Xw, Yw = [], []
    for xs, ds in zip(xs_list, ds_list):
        n = xs.shape[0]
        if n < seq_len:
            continue
        for i in range(0, n - seq_len + 1, stride):
            Xw.append(xs[i:i + seq_len])      # (T,1,H,W)
            Yw.append(ds[i + seq_len - 1])

    if not Xw:
        raise RuntimeError("No sliding windows generated. Check SEQ_LEN/STRIDE and data length.")

    Xw = np.stack(Xw, axis=0).astype(np.float32)  # (N,T,1,H,W)
    Yw = np.asarray(Yw, dtype=np.float32)         # (N,)
    return Xw, Yw


def save_npz(path, X, y):
    np.savez_compressed(path, X=X, y=y)


def main():
    rec_dirs = [
        os.path.join(DATA_ROOT, d)
        for d in sorted(os.listdir(DATA_ROOT))
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith('dca_angle_dist_record_')
    ]
    if not rec_dirs:
        raise RuntimeError(f"No 'angle_dist_record_*' folders found in {DATA_ROOT}.")

    xs_list, ds_list = [], []
    for rd in rec_dirs:
        xs0, ds = read_one_record_dir(rd)
        if xs0 is None or xs0.shape[0] < SEQ_LEN:
            continue
        xs_list.append(xs0)
        ds_list.append(ds)

    (tr_xs, tr_ds), (va_xs, va_ds), (te_xs, te_ds) = split_sequence_lists(xs_list, ds_list)

    X_tr, y_tr = build_windows(tr_xs, tr_ds, SEQ_LEN, STRIDE)
    X_va, y_va = build_windows(va_xs, va_ds, SEQ_LEN, STRIDE)
    X_te, y_te = build_windows(te_xs, te_ds, SEQ_LEN, STRIDE)

    mean_tr = X_tr.mean()
    std_tr = X_tr.std()
    if std_tr <= 0:
        raise RuntimeError("std computed as 0. Please check the data.")

    save_npz(os.path.join(OUT_DIR, 'train.npz'), X_tr, y_tr)
    save_npz(os.path.join(OUT_DIR, 'val.npz'), X_va, y_va)
    save_npz(os.path.join(OUT_DIR, 'test.npz'), X_te, y_te)

    with open(os.path.join(OUT_DIR, 'mean_std.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{mean_tr:.8f}\n{std_tr:.8f}\n")

    total_windows = X_tr.shape[0] + X_va.shape[0] + X_te.shape[0]
    print(f"Done. Windows: {total_windows}  | Train:{X_tr.shape[0]}  Val:{X_va.shape[0]}  Test:{X_te.shape[0]}")
    print(f"Sequences: Train:{len(tr_xs)}  Val:{len(va_xs)}  Test:{len(te_xs)}")
    print(f"[TRAIN] mean: {mean_tr:.6f}, std: {std_tr:.6f}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
