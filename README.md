[README.md](https://github.com/user-attachments/files/30072413/README.md)
# TI mmWave Radar Hand Tracking

本專案使用 **TI mmWave Radar**、**DCA1000EVM**、相機與 AprilTag，完成雷達資料擷取、角度與距離標註、模型訓練、即時手部追蹤及離線評估。

系統會將雷達資料轉換為 RDI、角度特徵等模型輸入，並透過 CNN + GRU 模型預測手部的角度與距離，進一步換算為 XY 座標，用於即時追蹤或虛擬滑鼠操作。

## 主要功能

- TI mmWave Radar 與 DCA1000EVM 原始資料擷取
- 相機與 AprilTag 角度、距離標註
- 角度與距離模型資料前處理
- CNN + GRU 模型訓練
- ZeroMQ 即時推論服務
- 雷達手部追蹤 GUI
- 手勢辨識與虛擬滑鼠控制
- 離線誤差與軌跡評估

## 專案流程

```text
Radar / DCA1000 設定
        ↓
錄製 Radar + Camera + AprilTag 標籤
        ↓
資料前處理與 Sliding Window 建立
        ↓
訓練角度與距離模型
        ↓
啟動 ZeroMQ 推論伺服器
        ↓
執行即時追蹤或離線評估
```

## 快速開始

### 1. 設定 Radar 與 DCA1000EVM

主要設定檔：

```text
xwr68xx_lvds_continuous.cfg
dca1000_config.json
```

請先確認：

- Radar 與 DCA1000EVM 網路設定正確
- Serial Port 與 Camera Index 正確
- 模型與資料路徑符合目前電腦環境

### 2. 錄製資料

DCA1000EVM 錄製工具：

```bash
python dca1000_rdi_raw_range_azimuth_camera.py
```

或使用 zero-Doppler angle FFT 版本：

```bash
python dca1000_rdi_zero_doppler_anglefft_camera.py
```

錄製結果通常會存放於：

```text
Record/dca_angle_dist_record_<timestamp>/
├── data_<timestamp>.h5
├── video_<timestamp>.mp4
└── records.csv
```

### 3. 資料前處理

角度資料：

```bash
python angle_data_preprocessing.py
```

距離資料：

```bash
python dist_data_preprocessing.py
```

處理後資料會存放於：

```text
Process_data/
├── Angle_data/
└── Distance_data/
```

### 4. 訓練模型

角度模型：

```bash
python angle_trainer.py
```

距離模型：

```bash
python dis_trainer.py
```

模型輸出位置：

```text
model_list/
├── angle_model/
└── dist_model/
```

### 5. 啟動即時推論

```bash
python inference.py
```

推論服務預設使用 ZeroMQ：

```text
tcp://localhost:5555
```

### 6. 執行即時追蹤

一般追蹤介面：

```bash
python mouse_tracking_v2.py
```

包含手勢辨識版本：

```bash
python mouse_tracking_gesture.py
```

### 7. 離線評估

```bash
python offline_eval_fixed_time_axis_with_metrics.py
```

主要輸出包含：

```text
offline_eval_predictions.csv
offline_eval_metrics_summary.txt
offline_eval_result.png
offline_eval_xy_result.png
```

## 主要檔案

| 檔案 | 用途 |
|---|---|
| `inference.py` | 載入角度、距離與手勢模型，提供 ZeroMQ 推論服務 |
| `mouse_tracking_v2.py` | 即時雷達手部追蹤 GUI |
| `mouse_tracking_gesture.py` | 結合手勢辨識的追蹤介面 |
| `angle_data_preprocessing.py` | 建立角度模型訓練資料 |
| `dist_data_preprocessing.py` | 建立距離模型訓練資料 |
| `angle_trainer.py` | 訓練角度回歸模型 |
| `dis_trainer.py` | 訓練距離回歸模型 |
| `offline_eval_fixed_time_axis_with_metrics.py` | 離線推論與誤差分析 |
| `calib_record.py` | 相機棋盤格校正 |
| `check_h5.py` | H5 雷達資料逐幀檢視 |

更完整的檔案說明請參考：

```text
PROJECT_FILES_OVERVIEW.md
```

## 資料夾結構

```text
.
├── Record/          # 原始錄製資料與評估結果
├── Process_data/    # 前處理後的訓練資料
├── model_list/      # 訓練完成的模型與指標
├── gesture_file/    # 手勢模型與相關資料
├── h5_snapshots/    # Radar heatmap 截圖
└── video_snapshots/ # 影片截圖
```

## 注意事項

- 執行前請先檢查各腳本中的資料路徑、COM Port、IP 與 Camera Index。
- 即時追蹤前需先啟動 `inference.py`。
- 模型輸入尺寸、正規化參數與錄製資料格式必須一致。
- 不同 Python 或 CUDA 環境可能需要重新安裝對應版本的 PyTorch 與相關套件。

## 技術內容

- Python
- PyTorch
- OpenCV
- PySide2
- ZeroMQ
- NumPy
- h5py
- TI mmWave Radar
- DCA1000EVM
- AprilTag

## License

本專案目前主要供研究與實驗使用。若需公開、修改或商業使用，請先確認相關模型、資料與第三方套件授權。
