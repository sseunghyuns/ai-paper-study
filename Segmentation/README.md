## 논문 리스트

### Segmentation
- [1] Fully Convolutional Networks for Semantic Segmentation | [논문](https://arxiv.org/abs/1411.4038), 설명, 구현 |
- [2] Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs | [논문](https://arxiv.org/abs/1412.7062), 설명, 구현 |
- [3] Multi-Scale Context Aggregation by Dilated Convolutions | [논문](https://arxiv.org/abs/1511.07122), 설명, 구현 |
- [4] Learning Deconvolution Network for Semantic Segmentation | [논문](https://arxiv.org/abs/1505.04366), 설명, 구현 |
- [5] Pyramid Scene Parsing Network | [논문](https://arxiv.org/abs/1612.01105), 설명, 구현 |
- [6] U-Net: Convolutional Networks for Biomedical Image Segmentation | [논문](https://arxiv.org/abs/1505.04597), [설명](#1), [구현](https://github.com/sseunghyuns/ai-paper-study/tree/main/paper_implementations/UNet) |
- [7] UNet++: A Nested U-Net Architecture for Medical Image Segmentation | [논문](https://arxiv.org/abs/1807.10165), 설명, 구현 |
- [8] UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation | [논문](https://arxiv.org/abs/2004.08790), 설명, 구현 |
- [9] Deep High-Resolution Representation Learning for Visual Recognition | [논문](https://arxiv.org/abs/1908.07919), 설명, 구현 |
- [10] UPSNet: A Unified Panoptic Segmentation Network | [논문](https://arxiv.org/abs/1901.03784), 설명, 구현 | 
- [11] Mask R-CNN | [논문](https://arxiv.org/abs/1703.06870), 설명, 구현 |
- [12] Path Aggregation Network for Instance Segmentation | [논문](https://arxiv.org/abs/1803.01534), 설명, 구현 |

---

## Quick paper reviews

### #1
#### U-Net: Convolutional Networks for Biomedical Image Segmentation

Fully-convolutional network로 이루어진 U-Net 구조를 제안하여 segmentation 분야에서 높은 성능을 달성하였다. U-Net은 이름에서도 알수 있다 싶이 U 형태의 네트워크 구조를 갖고 있다. Input 이미지가 들어오면 이미지의 특성을 추출하는 **contracting path**와 픽셀 단위로 예측을 하기 위해 다시 up-sampling 하는 **expansive path**가 존재한다. 일반적인 CNN 모델 구조와 달리 fully-connected layer가 존재하지 않는다. 

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/63924704/153896695-e6171970-55f8-4cff-9ede-37b782cd95f8.png">
</p>

</br>

저자들은 하나의 원본 이미지에서 여러 patch를 생성하여 모델의 입력값으로 주는 **overlap-tile strategy**을 적용하였고, 데이터 증강 기법으로는 **elastic deformation**을 사용하였다. 

<p align="center">
<img width="800" alt="스크린샷 2022-02-15 오전 12 46 00" src="https://user-images.githubusercontent.com/63924704/153896906-8f2d3b65-d204-4dec-a871-4244ef1af4f5.png">
</p>

</br>

또한 논문의 저자들은 객체 사이의 boundary를 더 잘 학습시키기 위해, boundary에 해당하는 pixel에 가중치(weight map)를 주는 방식을 사용했다.

<p align="center">
<img width="600" alt="1" src="https://user-images.githubusercontent.com/63924704/153897013-17c120b4-4ba7-4d9f-9468-dbf22e8e72c2.png">
</p>

</br>

U-Net은 biomedical 분야의 데이터를 분할하는 목적으로 제안됐지만, 현재 인용수가 20,000이 넘을 정도로 의료 영상뿐만 아니라 다양한 분야에서 활용되고 있다. 

보다 자세한 설명: [링크](https://seunghyun.oopy.io/9b26c0cf-9801-44e2-83c4-a8e5c880da23)

---
