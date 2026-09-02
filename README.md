# 回家作業 - 黃寀綾

## 安裝

執行：
`pip install -r requirements.txt`

## 資料下載

`requirements.txt`中已包含安裝 medmnist，或是執行：
`pip install -q medmnist`

執行`main.py`時，呼叫`dataset.py`下載 BloodMNIST 資料，可在`config.yaml`修改要下載資料集。

## 如何重跑每個實驗

執行：
`python main.py`
一次跑完所有實驗，包含評估。

## 如何調參

可在`config.yaml`修改參數。

## 預期執行時間與硬體
