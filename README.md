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

---

### Very Deep Convolutional Networks for Large-Scale Image Recognition

ImageNet Challenge 2014에서
- Increasing Depth
- Using small(3x3) convolution filters
