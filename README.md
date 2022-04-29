# Genesis_lab_task 김준수

## cifar10 + fishnet

- resize 128 : 경험적으로 224에 가깝게 resize 한 뒤에, 학습시키는게 성능이 좋았습니다.
- 5fold stratifiedfold로 일반화하려 했습니다.
- Augmix, test time augment 비교해보려 했습니다.
  - tta는 train_tranform으로 3번 augment해 주었습니다.
- cosineannealingwarmuprestart, warmup 3epoch, total 10epoch -> 사실상 cosineannealing처럼 사용했습니다.
- metric : acc1

## avg_acc1 result

train : 0.92
valid : 0.8280
test : 0.82640
tta : 0.82440

### 참고 repos, notebooks

- <https://github.com/kevin-ssy/FishNet>
- <https://github.com/yingmuying/Fishnet-PyTorch/tree/3ad047a82e389dfc219d2210dd049a61fe049a93> -> 사실상 모델구현적으로는 이 레포에서 거의 copy해왔습니다. 참가에 의의를 두겠습니다 ㅠㅠ
- <https://github.com/arabae/FishNet>
- augmi+albumentation : <https://www.kaggle.com/code/haqishen/augmix-based-on-albumentations/notebook>
- cifar10 + resnet : <https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min>
  - resnet9만으로도 10epoch 90%가 나오는 것을 보고 90프로를 목표로 했으나 잘되진 않았던 것 같습니다.
- tta : <https://www.kaggle.com/code/khyeh0719/pytorch-efficientnet-baseline-inference-tta>
