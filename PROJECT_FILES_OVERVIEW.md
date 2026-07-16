# TI Tracking 根目錄檔案說明

這份文件整理根目錄外層檔案的大致用途：每個檔案主要在做什麼、執行後會處理什麼資料，以及可能產生哪些輸出。整體專案大致是 TI mmWave radar / DCA1000 資料擷取、距離與角度標註、模型訓練、即時追蹤與離線評估流程。

## 典型流程

1. 使用 `.cfg` 與 `dca1000_config.json` 設定 radar / DCA1000。
2. 用錄製程式擷取 radar frame、camera video 與 AprilTag 標籤，產生 `Record/...` 底下的 `.h5`、`.mp4`、`records.csv`。
3. 用 `dist_data_preprocessing.py` / `angle_data_preprocessing.py` 將錄製資料切成訓練用 `.npz`。
4. 用 `dis_trainer.py` / `angle_trainer.py` 訓練距離與角度模型，輸出到 `model_list/...`。
5. 啟動 `inference.py`，讓其他程式透過 ZeroMQ 取得角度、距離或手勢推論結果。
6. 用 `mouse_tracking_v2.py` / `mouse_tracking_gesture.py` 做即時滑鼠追蹤。
7. 用 `offline_eval_fixed_time_axis_with_metrics.py` 與 `summarize_tracking_metrics_*.py` 產生評估圖、CSV 與總表。

## Python 腳本

| 檔案 | 大致用途 | 會做什麼 | 主要產物 |
|---|---|---|---|
| `angle_data_preprocessing.py` | 角度模型資料前處理 | 從 `Record/dca_angle_dist_record_*` 讀取 `.h5` 與 `records.csv`，取 `angle_deg` 作標籤，切 train/val/test，再建立長度 20 的 sliding window。 | `Process_data/Angle_data/train.npz`、`val.npz`、`test.npz`、`mean_std.txt` |
| `dist_data_preprocessing.py` | 距離模型資料前處理 | 從同一批錄製資料讀取 radar frame，取第 0 channel 作距離特徵，讀 `distance_cm` 作標籤，切 train/val/test 與 sliding window。 | `Process_data/Distance_data/train.npz`、`val.npz`、`test.npz`、`mean_std.txt` |
| `angle_trainer.py` | 訓練角度回歸模型 | 讀取 `Process_data/Angle_data` 的 `.npz`，用 CNN + GRU 訓練角度預測模型，計算 MAE、RMSE、R2、delta accuracy。 | `model_list/angle_model/<timestamp>/angle_model_seq.pth`、`angle_loss_curve.png`、`metrics.txt` |
| `dis_trainer.py` | 訓練距離回歸模型 | 讀取 `Process_data/Distance_data` 的 `.npz`，用 CNN + GRU 訓練距離預測模型。 | `model_list/dist_model/<timestamp>/dist_model_seq.pth`、`dist_loss_curve.png`、`metrics.txt` |
| `inference.py` | 即時推論伺服器 | 開 ZeroMQ REP server `tcp://*:5555`，載入角度模型、距離模型與手勢模型；接收 `angle`、`dist`、`both`、`gesture` 請求並回傳推論結果。 | 無固定檔案輸出，主要提供即時推論服務 |
| `mouse_tracking_v2.py` | 即時 radar 滑鼠追蹤介面 | 讀 DCA1000 即時 radar frame，送到 `inference.py` 推論角度/距離，轉成 XY 座標，在 PySide2 GUI 顯示追蹤點。 | 即時 GUI 顯示，通常不產生資料檔 |
| `mouse_tracking_gesture.py` | 加入手勢觸發的滑鼠追蹤介面 | 類似 `mouse_tracking_v2.py`，另外會送 gesture sequence 到推論伺服器，偵測到手勢時顯示觸發狀態並短暫 hold。 | 即時 GUI 顯示，通常不產生資料檔 |
| `dis_angle_recording.py` | UART demo parser + camera 標註錄製工具 | 透過 serial 傳 radar cfg，解析 TLV 中的 RDI 與 azimuth heatmap；同時用 AprilTag/camera 取得角度與距離標籤並錄製。 | `Record/angle_dist_record_<timestamp>/data_<timestamp>.h5`、`video_<timestamp>.mp4`、`records.csv` |
| `dca1000_rdi_raw_range_azimuth_camera.py` | DCA1000 raw radar + RDI/azimuth + camera 錄製工具 | 控制 DCA1000、送 radar cfg、接 UDP raw ADC，計算 range FFT / doppler FFT / azimuth map，顯示即時 OpenCV 視窗，也可按鍵錄製 radar frame 與 camera。 | `Record/dca_angle_dist_record_<timestamp>/data_<timestamp>.h5`、`video_<timestamp>.mp4`、`records.csv`、`last_model_input_4x64x32_AIC_range_pad2.npy` |
| `dca1000_rdi_zero_doppler_anglefft_camera.py` | zero-Doppler angle FFT 版本的 DCA1000 錄製工具 | 與上一支類似，但 azimuth map 偏向使用 zero/near-zero Doppler 的 angle FFT 處理方式。 | `Record/dca_angle_dist_record_<timestamp>/...`、`last_model_input_8x64x32_AIC_range_pad2.npy` |
| `python_dca1000_continuous_raw_AIC_realtime_RDI_OpenCV_pad2_top_third_integrated_controls_spaced.py` | 較早期/實驗版 DCA1000 即時 RDI 顯示 | 連續接收 DCA1000 raw ADC，做 AIC 背景抑制、RDI 計算與 OpenCV 即時控制介面；主要偏即時顯示與模型輸入確認。 | `last_model_input_8x64x32_AIC_range_pad2.npy` |
| `offline_eval_fixed_time_axis_with_metrics.py` | 單筆錄製資料離線評估 | 讀指定 `.h5` 與 `records.csv`，透過 `inference.py` 做逐 frame 預測，計算角度/距離/XY 誤差，並畫固定時間軸圖。 | 同資料夾的 `offline_eval_predictions.csv`、`offline_eval_metrics_summary.txt`、`offline_eval_result.png`、`offline_eval_xy_result.png` |
| `evaluate_from_h5.py` | 簡易離線評估腳本 | 對指定 `.h5` / `records.csv` 做角度與距離推論，計算 MAE/RMSE/R2，輸出角度/距離與 XY 圖。 | 根目錄的 `offline_eval_result.png`、`offline_eval_xy_result.png` |
| `summarize_tracking_metrics_ade_extra_py38.py` | 多受測者 tracking 指標總表 | 掃描 `Record/matt`、`ben`、`frank`、`u`、`jianhua` 底下的 `offline_eval_predictions.csv`，統計 MAE/SD、ADE、Frechet distance、TLS 等。 | `Record/tracking_metrics_summary_<timestamp>/` 底下的 CSV、Excel、`tracking_metrics_report.txt` |
| `summarize_tracking_metrics_ade_frechet_tls_maxpoint_py38.py` | 多受測者 tracking 指標總表，含最大對應點距離 | 與上一支相近，另外計算同時間點 prediction/GT 的最大歐氏距離，用於補充軌跡最大誤差。 | `Record/tracking_metrics_summary_<timestamp>/` 底下的 CSV、Excel、`tracking_metrics_report.txt` |
| `check_h5.py` | H5 radar frame 檢視器 | 讀指定 H5 的 `DS1`，以 OpenCV 顯示 RDI、azimuth、unshifted azimuth，支援逐 frame 查看與存圖。 | `h5_snapshots/frame_...png` |
| `half_h5.py` | H5 range 維度裁切工具 | 讀指定 H5 的 `DS1`，把最後一個 range 維度切成一半後另存。 | 原檔旁邊的 `*_range_half.h5` |
| `open_rdi_dat.py` | `.dat` TLV/RDI 解析與顯示工具 | 讀 TI UART/Demo Visualizer 類型的 `.dat`，找 magic word、解析 frame/TLV，取 type 5 Range-Doppler heatmap 並轉成可視化格式。 | 主要是畫圖/播放與終端輸出，無固定檔案輸出 |
| `calib_record.py` | 相機棋盤格校正工具 | 用 OpenCV 開 camera，按 `c` 擷取棋盤格畫面，收集足夠角點後計算相機內參與畸變。 | `calib_images/frame_*.png`、`calib.npz` |
| `video_frame_viewer.py` | 影片逐 frame 檢視器 | 讀指定影片，支援上一張/下一張、播放暫停、存 raw frame 或含 UI 的畫面。 | `video_snapshots/frame_...png` |

## 設定與資料檔

| 檔案 | 用途 | 內容/產物說明 |
|---|---|---|
| `xwr68xx_lvds_continuous.cfg` | DCA1000 LVDS raw capture 用 radar 設定 | 啟用 LVDS stream，常被 DCA1000 raw ADC 擷取腳本使用；目前設定 32 ADC samples、32 chirp loops、frame period 50 ms。 |
| `dca1000_config.json` | DCA1000 CLI 設定 | 定義 DCA1000 / PC IP、command/data port、raw LVDS capture、輸出路徑與檔案前綴等。 |
| `calib.npz` | 相機校正結果 | 由 `calib_record.py` 產生，包含 `K` 相機矩陣與 `dist` 畸變參數。 |
| `last_model_input_4x64x32_AIC_range_pad2.npy` | 最後一次 DCA1000 腳本儲存的模型輸入 | 用來快速檢查即時處理後送進模型的 radar feature，目前檔案 shape 為 `(4, 64, 32)`。 |
| `last_model_input_8x64x32_AIC_range_pad2.npy` | 另一版本最後模型輸入快照 | 用途同上，供 zero-Doppler / AIC 版本或實驗版腳本檢查輸入特徵，目前檔案 shape 為 `(4, 64, 32)`。 |

## 常見輸出資料夾

| 資料夾 | 內容 |
|---|---|
| `Record/` | 原始錄製資料、離線評估結果、跨受測者總表。 |
| `Process_data/` | 前處理後的 train/val/test `.npz` 與 mean/std。 |
| `model_list/` | 訓練好的距離/角度模型、loss curve、metrics。 |
| `gesture_file/` | 手勢模型、prototype、mean/std 與相關腳本。 |
| `h5_snapshots/` | `check_h5.py` 存出的 radar heatmap 圖。 |
| `video_snapshots/` | `video_frame_viewer.py` 存出的影片截圖。 |
