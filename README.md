# ai-paper-study

딥러닝 관련 논문들의 핵심 아이디어를 정리해놓은 저장소입니다.

---

## 논문 리스트

- [mixup: Beyond Empirical Risk Minimization](#1)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](#2)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](#cutMix-regularization-strategy-to-train-strong-classifiers-with-localizable-features)
- [Rethinking the Inception Architecture for Computer Vision](#rethinking-the-inception-architecture-for-computer-vision)
- [Deep Residual Learning for Image Recognition](#deep-residual-learning-for-image-recognition)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](#efficientNet-rethinking-model-scaling-for-convolutional-neural-networks)
- [Densely Connected Convolutional Networks](#densely-connected-convolutional-networks)
- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](#squeezenet-alexnet-level-accuracy-with-50x-fewer-parameters-and<0.5MB-model-size)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](#u-net-convolutional-networks-for-biomedical-image-segmentation)
- [UPSNet: A Unified Panoptic Segmentation Network](#upsnet-a-unified-panoptic-segmentation-network)
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](#faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks)
- [YOLOv4: Optimal Speed and Accuracy of Object Detection](#yolov4-optimal-speed-and-accuracy-of-object-detection)

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
