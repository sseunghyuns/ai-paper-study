# ai-paper-study

딥러닝 관련 논문들의 핵심 아이디어를 정리해놓은 저장소입니다.

---

## 논문 리스트

- [[1] mixup: Beyond Empirical Risk Minimization](#1)
- [[2] Very Deep Convolutional Networks for Large-Scale Image Recognition](#2)
- [[3] Deep Residual Learning for Image Recognition](#3)
- [[4] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](#4)
- Going Deeper with Convolutions
- CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
- Rethinking the Inception Architecture for Computer Vision
- Densely Connected Convolutional Networks
- SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- UPSNet: A Unified Panoptic Segmentation Network
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- YOLOv4: Optimal Speed and Accuracy of Object Detection

---

## Quick paper reviews

### #1
#### mixup: Beyond Empirical Risk Minimization

* 기존 Empirical Risk Minimization(ERM) 방법으로 학습된 크고 깊은 모델들은 강력하지만 adversarial examples에 대해 memorization(과적합)과 sensitivity의 문제를 보인다. 이에 대해 mixup , 즉 convex combinations 방식의 데이터 증강 기법을 적용하여 모델의 과적합을 줄이고 예측 강건함(robustness)를 높인다.

---

### #2
#### Very Deep Convolutional Networks for Large-Scale Image Recognition
VGGNet는 기존 `(Convolutional Layers → Pooling layers)의 반복 → Fully connected Layers` 의 전통적인 CNN 구조를 크게 벗어나지 않으면서, 레이어를 깊게 쌓아 2014 ILSVRC 이미지 분류 대회에서 2위를 달성하였다. (1위 GoogleNet)

VGGNet의 핵심은 기존 CNN에서 사용되었던 7x7, 5x5 크기의 필터들을 사용하지 않고 여러개의 3x3 필터로 쪼개어 사용함으로써 레이어를 더욱 깊게 쌓을 수 있었다는 것이다.  모든 Convolutional layers에서 비교적 작은 여러개의  3x3 필터만을 사용하게 되면 적은 파라미터 수로 깊게 레이어를 쌓을 수 있음과 동시에 여러 개의 비선형 함수를 사용할 수 있게 되므로 이를 통해 모델의 성능을 높일 수 있었다.

**3x3 필터 사용의 장점**

- 파라미터 수의 감소로 모델을 더욱 깊게 쌓을 수 있었다.
- 비선형성 증가로, 모델의 이미지 특징 식별성을 높인다.(Makes the decision function more discriminative.)

작은 필터 크기를 사용하여 모델의 깊이를 점차 늘려간다면, 파라미터의 수는 어쨌든 증가할 것이다. 이로 인해 한정된 학습 데이터에 대해 과적합 문제가 일어날 수 있고, 모델 깊이가 깊어짐에 따라 gradient vanishing/exploding 문제가 발생할 수 있을 것이다. 따라서 저자는 다음과 같은 기법을 사용하여 이러한 문제를 해결하였다고 한다.

- 과적합 문제→ Multi-Scale training(Scale Jittering)이라는 data augmentation 기법을 적용
- gradient 불안정 문제→얕은 모델에서 어느정도 학습된 가중치를 더욱 깊은 모델의 초기 가중치로 사용

정리하자면, VGGNet은 3x3 이라는 작은 필터 크기로 모델을 깊게 쌓아 학습을 진행하였고, 깊어진 모델로 인해 발생할 수 있는 과적합 문제와 gradient 불안정 문제를 각각 data augmentation과 가중치 초기화 전략으로 해결한 것이다.

VGGNet은 간단한 구조와, 단일 네트워크에서 GoogleNet보다 좋은 성능을 보여 지금까지도 많은 주목을 받고 있다.

---

### #3
#### Deep Residual Learning for Image Recognition
깊은 네트워크는 (1) gradient vanishing/exploding와 (2) degradation (of training accuracy)의 문제를 야기한다. (1)의 문제는 가중치 초기화 전략과 배치별 평균과 분산을 이용한 정규화 방법인 batch normalization 등을 통해 어느정도 해결할 수 있었다.

하지만 네트워크가 점점 더 깊어짐에 따라 (1)의 문제는 여전히 발생하였고, 학습 자체가 잘 안되는 (2)의 문제도 존재했다. 이를 해결하기 위해 본 논문에서 shortcut connection 기법을 통한 Residual learning을 제안했다. 아래는 Residual learning을 구현하는 하나의 residual block의 구조를 나타낸다.

<p align="center">
<img width="450" alt="1" src="https://user-images.githubusercontent.com/63924704/153001967-f7cdb834-3a5e-4850-a27d-83edc7183ad6.png">
</p>
  
Input x가 2개의 weight layers을 거친 후의 출력 결과를 <img src="https://latex.codecogs.com/svg.image?H(x)" title="H(x)" />라고 하자. 이때 모델은 학습을 통해 최적의 <img src="https://latex.codecogs.com/svg.image?H(x)" title="H(x)" /> 값을 찾아야 한다.  이때 모델이 기존의 unreferenced mapping인 <img src="https://latex.codecogs.com/svg.image?H(x)" title="H(x)" />를 학습하여 최적을 찾는 것보다, <img src="https://latex.codecogs.com/svg.image?F(x)&space;:=&space;H(x)-x" title="F(x) := H(x)-x" /> 를 학습하게 하여 더욱 쉽게 최적값을 찾을 수 있도록 하는 아이디어가 바로 residual learning인 것이다. 이때 기존의 출력 결과 <img src="https://latex.codecogs.com/svg.image?H(x)" title="H(x)" />는 <img src="https://latex.codecogs.com/svg.image?F(x)&space;&plus;&space;x" title="F(x) + x" />로 재정의된다. 

---

### #4
#### EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
Convolutional Neural Networks(ConvNets)에서 모델의 성능을 올리는 보편적인 방법은 네트워크의 **depth**(레이어의 깊이), **width**(채널 수), **resolution**(Input 이미지의 크기)를 키우는 것이다. 논문에서는 실험적으로 이 세 가지 요소를 모두 고려하여 네트워크의 크기를 키웠을 때 성능이 올라감을 보였다. 하지만 이러한 세 가지 요소는 상호 의존적이어서, 최적의 값을 찾기 위해 tuning하는 것에는 상당한 비용이 발생한다.

따라서 본 논문의 저자들은 compound scaling method를 제안하여 최적의 depth/width/resolution 조합을 tuning하는 방법론을 제안했고, 실제로 이 방법으로 ImageNet 데이터에서 SOTA를 달성했다. 이는 한정된 메모리와 모델의 성능(정확도)간의 trade-off를 고려한 scaling 방법이다. 

Compoind scaling method에서는 다음의 제약식을 만족하는 OOO을 탐색한다.

OOO

여기서 OOO는 그리드 서치를 통해 탐색하고, OOO는 한정된 자원 내에서 사용자가 정할 수 있는 하이퍼파라미터이다. 일반적으로 네트워크의 depth를 2배로 늘릴 경우 FLOPS도 2배 증가하지만, width나 resolution를 두배로 늘릴 경우 FLOPS는 4배 증가한다. 이를 반영한 제약식이 OOO인 것이다. OOO에 따라 네트워크의 FLOPS가 OOO로 증가하므로, 논문에서는 최대 OOO 정도까지만 FLOPS가 증가하도록 규제하였다.

또한 저자들은 EfficientNet 모델 구조를 통해 이러한 scaling 방법론의 성능을 확인하였다. EfficientNet은 MnasNet의 MBConv를 이용한 모델 구조로, scaling 정도에 따라 B0-B7까지 존재한다. 저자들은 EfficientNet-B0에서 찾은 OOO의 값을 B1~B7의 모델에 적용하였고, 각각의 모델들에서 최적의 OOO값을 찾았다고 한다. 

<p align="center">
<img width="450" alt="1" src="https://user-images.githubusercontent.com/63924704/153387089-8bcc0645-70c6-470d-823b-615ed07ee4be.png">
</p>


결론적으로 본 논문에서는 네트워크의 성능을 올릴 수 있는 최적의 width, depth, resolution 조합을 간단하면서 효율적으로 찾는 방법론을 제안한 것이다.

---




