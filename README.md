# Environment
跟作業的差不多

# train
```shell script=
bash run_train.sh
```

# inference
```shell script=
bash run_inference.sh
```

# 說明
* model 是改 EllSeg: An Ellipse Segmentation Framework for Robust Gaze Tracking 這篇github的RITnet_v7.py
* input是 192x256，把unet的output加了interpolation到 480x640
* 從unet中間的latent feature拉了一個classifier分類有沒有張眼睛
* loss部分用了很多大部分都是本來code裡有的下去改 (沒有非常確定有沒有改錯或理解錯)
* 我自己加了一個iou loss是用regression橢圓參數畫出mask再跟ground truth算iou
* 還有一個classification的cross entropy loss
* 橢圓的參數是用傳統那篇的code直接跑ground truth mask當input拿到的，在data.json裡面 (data2.json是角度沒有轉成sine的版本)
* 參數有中心點，長短軸跟順時針轉幾度，角度的部分我轉成sine來train，因為轉超過180度會一樣 (角度部分我沒有看到paper有這樣做，不確定正不正確)
* 角度要注意弧度跟角度的轉換

# 可能可以實驗的地方
* 把loss的比例調一調，有些loss可能根本沒用，或者有提升效果 (train_epoch跟val_epoch兩個function要一起改)
* 目前只有train在training set上，然後用validation set來驗證，目前這樣可以在20 epoch左右train到wiou: 0.9743, atnr: 0.9990, score: 0.9817，可以改成跑全部dataset下去train一次應該會更好
* data augmentation的部分目前沒做，有嘗試用過但還沒用好，主要是不知道橢圓的參數會因為旋轉或flip產生什麼變化
* 做data augmentation可以跑data.py，應該會生出可以看的圖

# 5/29 更新
* input改成240 * 320，但結果其實沒有太大變化
* 移除regression module，發現對結果沒有什麼影響
* loss 的部分，只剩下三個。移除的兩個都是regression module那邊的。
* 加了一些augmentation有讓結果有稍微提升。

# 之後可能方向
* 換其他model試試可能是transformer的等等
* 我沒試過分離classifier，不知道效果如何，另外classifier的cross entropy loss可能可以加weight看看，
* 後處理inference的圖片(可以用opencv的東西直接把output強制變橢圓)，因為我看了一下結果很多segmentation出來形狀有點怪有稍微試過幾張變橢圓看來結果有機會更好。
* 這份code還是沒有用全部dataset下去train，沒有validate的話可能要試試看要用哪種loss下去選。

* loss 的部分我查了滿多的，我覺得應該沒什麼可用的了。Accurate CNN-based pupil segmentation with an ellipse fit error regularization term 這篇github的loss我認為是不會work的，跟我之前寫的那個loss一樣的問題。