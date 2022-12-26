# 三維電腦視覺與深度學習應用-第十九組-SuperGlue: Learning Feature Matching with Graph Neural Networks

# 1.實驗一:SuperGlue在室内與室外效果的對比驗證<br />
負責同學:R10922172 彭旻翊<br />
放置資料夾:Exp_1<br />

## 實驗方式:<br />
一、針對indoor, outdoor不同場景使用不同model, 觀察效果<br />
二、調整並嘗試不同resize參數, 並觀察input size對結果的影響為何 <br />
三、調整並嘗試不同max_superpoints參數, 並觀察superpoints數量對結果的影響為何<br />
四、嘗試將indoor的圖片用於outdoor的model。outdoor的圖片用於indoor的model。觀察效果<br />

## 所需環境為:<br />
* Python 3 >= 3.5
* PyTorch >= 1.1
* OpenCV >= 3.4 (4.1.2.30 recommended for best GUI keyboard interaction, see this [note](#additional-notes))
* Matplotlib >= 3.1
* NumPy >= 1.18

## 如何執行:

The `--input` flag also accepts a path to a directory. We provide a directory of sample images.

```
./demo_superglue.py --input input/indoor/ --output_dir output/dump_indoor --resize 320 240 --no_display
```

The `--resize` flag can be used to resize the input image in three ways:

1. `--resize` `width` `height` : will resize to exact `width` x `height` dimensions
2. `--resize` `max_dimension` : will resize largest input image dimension to `max_dimension`
3. `--resize` `-1` : will not resize (i.e. use original image dimensions)
4. `--max_keypoints` : Maximum number of keypoints detected by Superpoint ('-1' keeps all keypoints) (default: -1)
5. Use `--show_keypoints` to visualize the detected keypoints (default: `False`).


# 2.實驗二:Physarum Dynamics 結合 SuperGlue <br />
負責同學:R11922185 杜嘉煒<br />
放置資料夾:Exp_3_2<br />

## 所需環境為:<br />
* Python == 3.8.10 
* Torch == 1.13.0a0+936e930
* OpenCV_Python == 4.5.1
* matplotlib == 3.6.2
* NumPy == 1.23.5

## 在使用過程中可能會出現：

libGL.so.1: cannot open shared object file: No such file or directory 的報錯，我們只需執行以下命令即可修復

```

sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y

```
## 如何執行:
```
python trainer.py 
```

## Additional useful command line parameters
`--epoch` 可以設置epoch數目 (default: `20`).

`--train_path` 用來設置你的訓練資料集的位置.

`--eval_output_dir` 用來設置可視化的圖片的輸出位置 (default: `pairs/`).

`--show_keypoints` 可以設置是否查看關鍵點的位置 (default: `False`).

## 更換訓練集:

若要更換爲自己的訓練集，需要命名為`COCO`+`數字`的組合，也可以直接執行rename.py

# 3.實驗三3.1- Visual Odometry的比較
負責同學:R11943113 葉冠宏<br />
放置資料夾:Exp_3_1<br />

## 說明:
本次實驗主要在比較superpoint+superglue用於作業三之效果呈現。由於原本作業三已經執行過orb+brute-force matcher，因此資料夾的code中僅執行superpoint+superglue的部分。<br />

## 如何執行:
```
python vo.py  --input ./frames/ --camera_parameters camera_parameters.npy
```
--input 後面放置的參數為影像所放置的資料夾，--camera_parameters 後面所放置的參數為相機的內在參數檔案。


## 所使用的環境:<br />
* numpy
* cv2
* sys
* os
* argparse
* glob
* torch
* collections
* matplotlib
* copy
* pathlib
* open3d

# 4.實驗三3.2- trajectory 的比較
負責同學:R11943113 葉冠宏<br />
放置資料夾:Exp_3_2<br />

## 說明:
本次實驗主要在比較orb+bruteforce matcher, superpoint+flann matcher, superpoint+superglue 三種方法對於trajectory的預測情形，以及和groundtruth做比較。我使用的資料集是來自於KITTI。<br />

## 如何執行:
```
python trajectory.py  --detector superpoint --matcher superglue
```
<br />
--detector 後面所放的參數為所欲使用的detector演算法，包含superpoint或是ORB。--matcher 後面所放置的參數為所欲使用的matching演算法，包含superglue,flann,或是bruteforce matcher。<br />

## 所使用的環境:
* numpy
* cv2
* sys
* os
* argparse
* glob
* torch
* collections
* matplotlib
* copy
* pathlib

