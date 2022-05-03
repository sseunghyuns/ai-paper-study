## 논문 리스트

---

### Classification
- [1] Very Deep Convolutional Networks for Large-Scale Image Recognition | [논문](https://arxiv.org/abs/1409.1556), [설명](#1), 구현 |
- [2] Deep Residual Learning for Image Recognition | [논문](https://arxiv.org/abs/1512.03385), [설명](#2), [구현](https://github.com/sseunghyuns/ai-paper-study/tree/main/paper_implementations/ResNet) |
- [3] Densely Connected Convolutional Networks | [논문](https://arxiv.org/abs/1608.06993), [설명](#3), 구현 |
- [4] SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size | [논문](https://arxiv.org/abs/1602.07360), [설명](#4), 구현 |
- [5] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks | [논문](https://arxiv.org/abs/1905.11946), [설명](#5), 구현 | 
- [6] Going Deeper with Convolutions | [논문](https://arxiv.org/abs/1409.4842), 설명, 구현 | 
- [7] Rethinking the Inception Architecture for Computer Vision | [논문](https://arxiv.org/abs/1512.00567), 설명, 구현 | 

---

## Quick paper reviews

### #1
#### Very Deep Convolutional Networks for Large-Scale Image Recognition

VGGNet는 기존 `(Convolutional Layers → Pooling layers)의 반복 → Fully connected Layers` 의 전통적인 CNN 구조를 크게 벗어나지 않으면서, 레이어를 깊게 쌓아 2014 ILSVRC 이미지 분류 대회에서 2위를 달성하였다. (1위 GoogleNet)

<p align="center">
<img width="500" alt="1" src="https://user-images.githubusercontent.com/63924704/153600904-93852b64-c4ba-4e8c-926e-60bc190a0701.png">
</p>

VGGNet의 핵심은 기존 CNN에서 사용되었던 7x7, 5x5 크기의 필터들을 사용하지 않고 여러개의 3x3 필터로 쪼개어 사용함으로써 레이어를 더욱 깊게 쌓을 수 있었다는 것이다.  모든 Convolutional layers에서 비교적 작은 여러개의  3x3 필터만을 사용하게 되면 적은 파라미터 수로 깊게 레이어를 쌓을 수 있음과 동시에 여러 개의 비선형 함수를 사용할 수 있게 되므로 이를 통해 모델의 성능을 높일 수 있었다.

**3x3 필터 사용의 장점**

- 파라미터 수의 감소로 모델을 더욱 깊게 쌓을 수 있었다.
- 비선형성 증가로, 모델의 이미지 특징 식별성을 높인다.(Makes the decision function more discriminative.)

작은 필터 크기를 사용하여 모델의 깊이를 점차 늘려간다면, 파라미터의 수는 어쨌든 증가할 것이다. 이로 인해 한정된 학습 데이터에 대해 과적합 문제가 일어날 수 있고, 모델 깊이가 깊어짐에 따라 gradient vanishing/exploding 문제가 발생할 수 있을 것이다. 따라서 저자는 다음과 같은 기법을 사용하여 이러한 문제를 해결하였다고 한다.

- 과적합 문제→ Multi-Scale training(Scale Jittering)이라는 data augmentation 기법을 적용
- gradient 불안정 문제→얕은 모델에서 어느정도 학습된 가중치를 더욱 깊은 모델의 초기 가중치로 사용

정리하자면, VGGNet은 3x3 이라는 작은 필터 크기로 모델을 깊게 쌓아 학습을 진행하였고, 깊어진 모델로 인해 발생할 수 있는 과적합 문제와 gradient 불안정 문제를 각각 data augmentation과 가중치 초기화 전략으로 해결한 것이다.

VGGNet은 간단한 구조와, 단일 네트워크에서 GoogleNet보다 좋은 성능을 보여 지금까지도 많은 주목을 받고 있다.

보다 자세한 설명: [링크](https://seunghyun.oopy.io/f8970a0f-246a-4edf-8472-719ea22b179f)

---

### #2
#### Deep Residual Learning for Image Recognition

깊은 네트워크는 (1) gradient vanishing/exploding와 (2) degradation (of training accuracy)의 문제를 야기한다. (1)의 문제는 가중치 초기화 전략과 배치별 평균과 분산을 이용한 정규화 방법인 batch normalization 등을 통해 어느정도 해결할 수 있었다.

하지만 네트워크가 점점 더 깊어짐에 따라 (1)의 문제는 여전히 발생하였고, 학습 자체가 잘 안되는 (2)의 문제도 존재했다. 이를 해결하기 위해 본 논문에서 shortcut connection 기법을 통한 Residual learning을 제안했다. 아래는 Residual learning을 구현하는 하나의 residual block의 구조를 나타낸다.

<p align="center">
<img width="450" alt="1" src="https://user-images.githubusercontent.com/63924704/153001967-f7cdb834-3a5e-4850-a27d-83edc7183ad6.png">
</p>
  
Input x가 2개의 weight layers을 거친 후의 출력 결과를 <img src="https://latex.codecogs.com/svg.image?H(x)" title="H(x)" />라고 하자. 이때 모델은 학습을 통해 최적의 <img src="https://latex.codecogs.com/svg.image?H(x)" title="H(x)" /> 값을 찾아야 한다.  이때 모델이 기존의 unreferenced mapping인 <img src="https://latex.codecogs.com/svg.image?H(x)" title="H(x)" />를 학습하여 최적을 찾는 것보다, <img src="https://latex.codecogs.com/svg.image?F(x)&space;:=&space;H(x)-x" title="F(x) := H(x)-x" /> 를 학습하게 하여 더욱 쉽게 최적값을 찾을 수 있도록 하는 아이디어가 바로 residual learning인 것이다. 이때 기존의 출력 결과 <img src="https://latex.codecogs.com/svg.image?H(x)" title="H(x)" />는 <img src="https://latex.codecogs.com/svg.image?F(x)&space;&plus;&space;x" title="F(x) + x" />로 재정의된다. 

---

### #3
#### Densely Connected Convolutional Networks

ResNet의 skip-connection 구조와 유사하게, 각각의 레이어가 다른 모든 레이어와 연결되어있는 DenseNet 구조를 제안하여 여러 오픈 데이터셋에서 SOTA 성능을 달성하였다. 레이어간 연결을 최대화하여 feature 정보를 최대한 활용하겠다는 아이디어이다.

DenseNet은 이전 레이어의 정보를 현재 레이블에 반영한다는 점에서 ResNet과 유사하지만, summation을 하는 ResNet과 달리 (channel-wise)concatenation을 사용한다. 

<p align="center">
<img width="500" alt="1" src="https://user-images.githubusercontent.com/63924704/153897707-2b9227db-c6e3-41b3-ae3d-cc84c5a9106b.png">
</p>

위 그림은 DenseNet을 이루는 하나의 Dense Block이다. 여기서 <img src="https://render.githubusercontent.com/render/math?math=x_l">는 <img src="https://render.githubusercontent.com/render/math?math=l">번째 레이어를 통과하는 feature이고, <img src="https://render.githubusercontent.com/render/math?math=H_l">은 (Batch Normalization → ReLU → 3x3 Conv)로 구성된 composite function이다. 

이러한 Dense Block이 여러개 모여 하나의 DenseNet을 이룬다(아래 그림 참고). 이때, growth rate <img src="https://render.githubusercontent.com/render/math?math=k">는 하이퍼 파라미터로써 Dense Block 내의 채널 수를 조절한다. 

<p align="center">
<img width="800" alt="1" src="https://user-images.githubusercontent.com/63924704/153898157-0d0ed053-f84b-4371-a4f0-ba292f92fc50.png">
</p>

Dense Block의 마지막 레이어에서 나오는 feature map의 사이즈는 concatenation으로 늘어난 상태이다. 이를 downsampling하기 위해 두 Dense Block 사이에 (Batch Normalization → 1x1 Conv → 2x2 Average Pooling)으로 구성된 transition layer를 추가하였다. 

이러한 네트워크 구조를 사용했을 때의 장점은 다음과 같다.  
1. Elleviate the vanishing-gradient problem
2. Strengthen feature propagation
3. Encourage feature reuse
4. Reduce the number of parameters
5. Parameter efficiency(더 적은 파라미터수로 ResNet보다 좋은 성능을 냈다)

---

### #4
#### SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size

AlexNet 수준의 성능을 보이면서도 50배 적은 파라미터 수를 가진 SqueezeNet 구조를 제안하였다. 네트워크가 적은 파라미터 수를 가졌을 때의 장점은 다음과 같다.

1. More efficient distributed training
2. Less overhead when exporting new models to clients
3. Feasible FPGA and embedded deployment

</br>

AlexNet과 비슷한 성능을 보이면서 훨씬 적은 파라미터 수를 갖도록 하기 위해 저자들은 `Fire module` 을 만들어 적용했다. 이러한 아키텍처는 아래 세 가지 전력을 기반으로 한다.  


- **Strategy 1.** Replace 3x3 filters with 1x1 filters. 
  
  → 3x3보다 1x1필터가 9배 적은 파라미터 수를 갖고 있으므로, 이를 통해 파라미터 수를 줄이는 것이 가능하다.


- **Strategy 2.** Decrease the number of input channels to 3x3 filters. 
  
  → 3x3 크기의 필터만을 사용하는 네트워크가 있을 때, 특정 레이어의 파라미터 수는 (number of input channels) * (number of filters) * (3*3)으로 계산된다. 여기서 strategy 1을 통해 3x3 필터를 1x1 필터로 바꾸는 것 뿐만 아니라 input channels 역시 줄여서 파라미터 수를 감소하겠다는 것이다.


- **Strategy 3.** Downsample late in the network so that convolution layers have activation maps. 
  
  → 최대한 downsampling을 네트워크 뒷단에서 이루어지게 하여, 큰 activation map 정보를 계속 유지할 수 있도록 한다.
  
</br>

##### Fire Module

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/63924704/154033473-5a4a5d3b-2cd6-404e-ae9c-cb98f4b4d7ca.png">
</p>

- `squeeze layer` 와 `expand layer` 로 이루어져 있다.  
- `squeeze layer` 는 오직 1x1 conv layers, `expand layer` 는 1x1와 3x3 conv layers의 조합으로 구성된다.  
  - <img src="https://render.githubusercontent.com/render/math?math=s_{1x1}">: squeeze layer에 있는 1x1 필터의 수
  - <img src="https://render.githubusercontent.com/render/math?math=e_{1x1}">: expand layer에 있는 1x1 필터의 수
  - <img src="https://render.githubusercontent.com/render/math?math=e_{3x3}">: expand layer에 있는 3x3 필터의 
- <img src="https://render.githubusercontent.com/render/math?math=s_{1x1} < (e_{1x1} + e_{3x3})"> 으로 설정하여 input channels를 조절하였다.(Strategy 2)

</br>

<p align="center">
<img width="700" alt="1" src="https://user-images.githubusercontent.com/63924704/154034202-84a6d52c-ee40-4666-96c1-a70b99425ece.png">
</p>

SqueenzeNet 구조를 보여주는 그림이다. Strategy 1,2를 따르는 8개의 Fire modules를 사용하였고, Strategy 3에 따라 max-pooling layer가 상대적으로 늦게 적용되고 있다. 

---

### #5
#### EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

Convolutional Neural Networks(ConvNets)에서 모델의 성능을 올리는 보편적인 방법은 네트워크의 **depth**(레이어의 깊이), **width**(채널 수), **resolution**(Input 이미지의 크기)를 키우는 것이다. 논문에서는 실험적으로 이 세 가지 요소를 모두 고려하여 네트워크의 크기를 키웠을 때 성능이 올라감을 보였다. 하지만 이러한 세 가지 요소는 상호 의존적이어서, 최적의 값을 찾기 위해 tuning하는 것에는 상당한 비용이 발생한다.

따라서 본 논문의 저자들은 **compound scaling method**를 제안하여 최적의 depth/width/resolution 조합을 tuning하는 방법론을 제안했고, 실제로 이 방법으로 ImageNet 데이터에서 SOTA를 달성했다. 이는 한정된 메모리와 모델의 성능(정확도)간의 trade-off를 고려한 scaling 방법이다. 

Compoind scaling method에서는 다음의 제약식을 만족하는 <img src="https://render.githubusercontent.com/render/math?math=d,w,r">을 탐색한다.

<p align="center">
<img width="225" alt="스크린샷 2022-02-11 오후 10 23 30" src="https://user-images.githubusercontent.com/63924704/153599019-d028b458-af20-4795-8d2b-448229c3eca3.png">
</p>

여기서 <img src="https://render.githubusercontent.com/render/math?math=d,w,r">는 그리드 서치를 통해 탐색하고, <img src="https://render.githubusercontent.com/render/math?math=\phi">는 한정된 자원 내에서 사용자가 정할 수 있는 하이퍼파라미터이다. 일반적으로 네트워크의 depth를 2배로 늘릴 경우 FLOPS도 2배 증가하지만, width나 resolution를 두배로 늘릴 경우 FLOPS는 4배 증가한다. 이를 반영한 제약식이 <img src="https://render.githubusercontent.com/render/math?math=\alpha \beta^2 \gamma^2 \approx 2">인 것이다. <img src="https://render.githubusercontent.com/render/math?math=\phi">에 따라 네트워크의 FLOPS가 <img src="https://render.githubusercontent.com/render/math?math=(\alpha \beta^2 \gamma^2)^{\phi}">로 증가하므로, 논문에서는 최대 <img src="https://render.githubusercontent.com/render/math?math=2^{\phi}"> 정도까지만 FLOPS가 증가하도록 규제하였다.

또한 저자들은 EfficientNet 모델 구조를 통해 이러한 scaling 방법론의 성능을 확인하였다. EfficientNet은 MnasNet의 MBConv를 이용한 모델 구조로, scaling 정도에 따라 B0-B7까지 존재한다. 저자들은 EfficientNet-B0에서 찾은 <img src="https://render.githubusercontent.com/render/math?math=\alpha=1.2, \beta=1.1, \gamma=1.15">의 값을 B1~B7의 모델에 적용하였고, 각각의 모델들에서 최적의  <img src="https://render.githubusercontent.com/render/math?math=\phi">값을 찾았다고 한다. 

<p align="center">
<img width="450" alt="1" src="https://user-images.githubusercontent.com/63924704/153387089-8bcc0645-70c6-470d-823b-615ed07ee4be.png">
</p>


결론적으로 본 논문에서는 네트워크의 성능을 올릴 수 있는 최적의 width, depth, resolution 조합을 간단하면서 효율적으로 찾는 방법론을 제안한 것이다.

---

