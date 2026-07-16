import h5py
import numpy as np
import os

input_path = r"C:\Users\mc2\Desktop\ti_tracking\Record\angle_dist_record_20260428_111626\data_20260428_111626.h5"
output_path = input_path.replace(".h5", "_range_half.h5")

with h5py.File(input_path, "r") as fin:
    data = fin["DS1"][:]   # shape: (N, C, Doppler, Range)

    print("Original shape:", data.shape)

    # 砍掉 range 後半，只保留前半
    range_bins = data.shape[-1]
    half_range = range_bins // 2
    data_half = data[:, :, :, :half_range]

    print("New shape:", data_half.shape)

    with h5py.File(output_path, "w") as fout:
        fout.create_dataset("DS1", data=data_half, dtype=np.float32)

print("Saved to:", output_path)