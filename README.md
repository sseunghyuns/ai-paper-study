# ai-paper-study

딥러닝 관련 논문들의 핵심 아이디어를 정리해놓은 저장소입니다.

---

## 논문 리스트

- [mixup: Beyond Empirical Risk Minimization](#mixup-beyond-empirical-risk-minimization)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](#cutMix-regularization-strategy-to-train-strong-classifiers-with-localizable-features)
- [Rethinking the Inception Architecture for Computer Vision](#rethinking-the-inception-architecture-for-computer-vision)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](#very-deep-convolutional-networks-for-large-scale-image-recognition)
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

### mixup: Beyond Empirical Risk Minimization

* 기존 Empirical Risk Minimization(ERM) 방법으로 학습된 크고 깊은 모델들은 강력하지만 adversarial examples에 대해 memorization(과적합)과 sensitivity의 문제를 보인다. 이에 대해 mixup , 즉 convex combinations 방식의 데이터 증강 기법을 적용하여 모델의 과적합을 줄이고 예측 강건함(robustness)를 높인다.

**What is ...**
<details>
<summary> adversarial examples?</summary>
<div markdown="1">
 
- Data just outside the training distribution.
- 신경망을 혼란시킬 목적으로 만들어진 특수한 입력으로, 신경망으로 하여금 샘플을 잘못 분류하도록 한다. 비록 인간에게 적대적 샘플은 일반 샘플과 큰 차이가 없어보이지만, 신경망은 적대적 샘플을 올바르게 식별하지 못한다.  - [Tensorflow](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm?hl=ko)
  
</div>
</details>

<details>
<summary> memorization?</summary>
<div markdown="1">
 
- 모델이 훈련시 학습 데이터를 기억 하는 것. 따라서 학습 데이터 분포 외의 데이터를 만나면, 예측을 잘 하지 못하는 문제가 발생한다. 
- 과적합과 같은 말인 것 같다. 

</div>
</details>

</div>
</details>

<details>
<summary>Empirical Risk Minimization(ERM)?</summary>
<div markdown="1">
 
- The idea is that we don’t know exactly how well an algorithm will work in practice (the true "risk") because we don't know the true distribution of data that the algorithm will work on but as an alternative we can measure its performance on a known set of training data. - [kaggle](https://www.kaggle.com/general/262639)
- 모집단이 아닌 표본 집단(우리가 현재 보유한 데이터)을 통해 ‘경험적’인 Risk(loss)를 최소화하는 방법이다. 즉, 우리가 가지고 있는 데이터에 대한 risk를 최소화하는 것이다.
- 주어진 한정된 훈련 데이터의 분포를 따르는 손실함수의 기대값을 최소화하는 값을 찾는 과정

</div>
</details>

<details>
<summary>Vicinal Risk Minimization(VRM)?</summary>
<div markdown="1">
 
- 훈련 데이터 + 훈련 데이터 근방(vicinal)의 분포까지도 학습하여 추가적인 data에 대해 결론 도출이 가능해진다.

</div>
</details>

</div>
</details>

<details>
<summary>gradient norms?</summary>
<div markdown="1">
 
- 그레디언트의 크기
- 오류가 많으면, 그레디언트 기울기가 가팔라지므로, 그 크기가 커지게 된다.
  
</div>
</details>

<details>
<summary>convex combination?</summary>
<div markdown="1">
 
- 주어진 지점을 서로 연결한 도형 안에 존재하는 지점들 (참고 [링크](https://light-tree.tistory.com/176))
- 모델 학습으로 생각해보면, 학습시 데이터 증강을 통해 새로운 데이터를 학습에 사용한다는 의미이다.
  
</div>
</details>
