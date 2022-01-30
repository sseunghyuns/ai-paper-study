# ai-paper-study

딥러닝 관련 논문들의 핵심 아이디어를 정리해놓은 저장소입니다.

---

## 논문 리스트

- [mixup: Beyond Empirical Risk Minimization](#mixup: Beyond Empirical Risk Minimization)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](#no2)
- [Rethinking the Inception Architecture for Computer Vision](#no3)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](#no4)
- [Deep Residual Learning for Image Recognition](#no5)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](#no6)
- [Densely Connected Convolutional Networks](#no7)
- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](#no8)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](#no9)
- [UPSNet: A Unified Panoptic Segmentation Network](#no10)
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](#no11)
- [YOLOv4: Optimal Speed and Accuracy of Object Detection](#no12)

---

## Quick paper reviews

### mixup: Beyond Empirical Risk Minimization

* 기존 Empirical Risk Minimization(ERM) 방법으로 학습된 크고 깊은 모델들은 강력하지만 adversarial examples에 대해 memorization(과적합)과 sensitivity의 문제를 보인다. 이에 대해 mixup , 즉 convex combinations 방식의 데이터 증강 기법을 적용하여 모델의 과적합을 줄이고 예측 강건함(robustness)를 높인다.
