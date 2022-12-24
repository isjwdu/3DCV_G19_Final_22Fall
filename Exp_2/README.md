# README

## 所需環境為
Python == 3.8.10 

Torch == 1.13.0a0+936e930

OpenCV_Python == 4.5.1

matplotlib == 3.6.2

NumPy == 1.23.5

在使用過程中可能會出現：

libGL.so.1: cannot open shared object file: No such file or directory 的報錯，我們只需執行以下命令即可修復

```

sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y

```
## 如何執行
```
python trainer.py 
```

### Additional useful command line parameters
`--epoch` 可以設置epoch數目 (default: `20`).

`--train_path` 用來設置你的訓練資料集的位置.

`--eval_output_dir` 用來設置可視化的圖片的輸出位置 (default: `pairs/`).

`--show_keypoints` 可以設置是否查看關鍵點的位置 (default: `False`).

