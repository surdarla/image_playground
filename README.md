# Genesis_lab_task

## cifar10 + fishnet
- resize 128 : 경험적으로 사이즈를 키워서 학습시키는게 성능이 좋았습니다
- 5fold stratifiedfold로 일반화하려 했습니다.
- Augmix, test time augment 비교 
- cosineannealingwarmuprestart, warmup 3epoch

### 참고 repo

- https://github.com/kevin-ssy/FishNet
- https://github.com/yingmuying/Fishnet-PyTorch/tree/3ad047a82e389dfc219d2210dd049a61fe049a93 -> 사실상 모델구현적으로는 이 레포에서 거의 copy해왔습니다.
- https://github.com/arabae/FishNet