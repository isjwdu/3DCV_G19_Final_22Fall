# README

#1.實驗一:SuperGlue在室内與室外效果的對比驗證<br />
負責同學:R10922172 彭旻翊<br />

實驗方式:<br />
一、indoor的圖片用於outdoor的model。outdoor的圖片用於indoor的model。<br />
二、調整resize參數<br />
三、調整max_superpoints參數<br />

#2.實驗二:Physarum Dynamics 結合 SuperGlue <br />
負責同學:R11922185 杜嘉煒<br />
<br />

所需環境為:<br />
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
如何執行:
```
python trainer.py 
```

Additional useful command line parameters
`--epoch` 可以設置epoch數目 (default: `20`).

`--train_path` 用來設置你的訓練資料集的位置.

`--eval_output_dir` 用來設置可視化的圖片的輸出位置 (default: `pairs/`).

`--show_keypoints` 可以設置是否查看關鍵點的位置 (default: `False`).

更換訓練集:

若要更換爲自己的訓練集，需要命名為`COCO`+`數字`的組合，也可以直接執行rename.py

#3.實驗三3.1- Visual Odometry的比較
負責同學:R11943113 葉冠宏
<br />
<br />
放置資料夾:Exp_3_1
<br />
說明:
本次實驗主要在比較superpoint+superglue用於作業三之效果呈現。由於原本作業三已經執行過orb+brute-force matcher，因此資料夾的code中僅執行superpoint+superglue的部分。<br />

如何執行:
```
python vo.py  --input ./frames/ --camera_parameters camera_parameters.npy
```
--input 後面放置的參數為影像所放置的資料夾，--camera_parameters 後面所放置的參數為相機的內在參數檔案。


所使用的環境:<br />
numpy<br />
cv2<br />
sys<br />
os<br />
argparse<br />
glob<br />
torch<br />
collections<br />
matplotlib<br />
copy<br />
pathlib<br />
open3d<br />




#4.實驗三3.2- trajectory 的比較
負責同學:R11943113 葉冠宏
<br />
<br />
放置資料夾:Exp_3_2

<br />
說明:
本次實驗主要在比較orb+bruteforce matcher, superpoint+flann matcher, superpoint+superglue 三種方法對於trajectory的預測情形，以及和groundtruth做比較。我使用的資料集是來自於KITTI。<br />

如何執行:
```
python trajectory.py  --detector superpoint --matcher superglue
```
<br />
--detector 後面所放的參數為所欲使用的detector演算法，包含superpoint或是ORB。--matcher 後面所放置的參數為所欲使用的matching演算法，包含superglue,flann,或是bruteforce matcher。<br />
所使用的環境:<br />

numpy<br />
cv2<br />
sys<br />
os<br />
argparse<br />
glob<br />
torch<br />
collections<br />
matplotlib<br />
copy<br />
pathlib<br />

