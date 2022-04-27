# Genesis_lab_task

fishnet

1. direct BP
   * gradient from the deep layer ->(directly propagated) shallow layers
   * identifiy mapping with residual block + concatenation 로 resnet이 보여줬으나,
   * this problem is cause by the conv layer between features of different resolutions 다른 해상도에서의 feature들끼리 conv later들이 인식?받지를 못한다?
   * convolution without direct bp -> degrade the gradient from the output to shallow layers
   * solution - concatenating features of different depths to the final output
   * semantic meaning of features 도 보존된다.

2. features of different depths preserved used for refining each other
    * features with different depths have different levels of abstraction of img
    * trade-off(num of parameters vs acc of classification)문제는?
    * complementary?
    * Mask R-CNN?

**related works**

>tail - existing works, to obtain deep low-resolution features
>body - get high-resolution features of high-level semantic info = *****
>head - preserve+refine the features from three parts. concat!!!

1. cnn arch
   1. alexnet vgg inception -> deeper -> gradient vanishing
   2. the features of high resolution are extracted by the shallow layers with small receptive field -> lack of high-level semantic meaning
   3. we are first of 'extract high-resolution deep feature with high-level semantic meaning'
2. combine features from diff resolution,depth <- nested sparse net,hyper col,addition,residual blocks
3. up-sampling - large feature maps to keep resolution이 필요.
   unet이랑 비슷하다.
   MSDNet은 비슷하긴한데, 다른해상도의 feature들끼리의 사용해서 여전히 classification에서는 별로다.
4. message passing among features.outputs

**identity mapping in deep residual networks and isolated conv**

in resnet - the features of diff resolution are diff in num of channels. -> transion func h needed before down-sampling
* gradient propagation problem from *i-conv*
    * residual block with identity mapping(resnet), dense block(densenet) = dircet gradient propagation
    * 다른 해상도간의 i-conv(resnet), 인접한 dense block 간의 i-conv -> DGP안됨
    * invertible resent은 현재 feature 전부를 다음 스테이지로 보냈는데 parameter explode

## Fishnet

UR-block : up-sampling  + refinement block
DR-block : down-sampling + refinement block
    UR이랑 다른 점
    1. 2x2 max-pooling for down-sampling
    2. no channel reduction func(r) -> 이래서 사실상 DR-block == residual-block

* head is composed of concatenations,no i-conv, no conv with identity mapping, no max-pooling
* down-sampling kennel-size : 2x2, stride = 2 -> avoid overlapping between pixel
* 