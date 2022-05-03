## 논문 리스트

### Object Detection
- [1] Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | [논문](https://arxiv.org/abs/1506.01497), [설명](#1), 구현 |
- [2] Feature Pyramid Networks for Object Detection | [논문](https://arxiv.org/abs/1612.03144), 설명, 구현 |
- [3] SSD: Single Shot MultiBox Detector | [논문](https://arxiv.org/abs/1512.02325), 설명, 구현 |
- [4] Focal Loss for Dense Object Detection | [논문](https://arxiv.org/abs/1708.02002), 설명, 구현 |
- [5] Cascade R-CNN: Delving into High Quality Object Detection | [논문](https://arxiv.org/abs/1712.00726), 설명, 구현 |
- [6] You Only Look Once: Unified, Real-Time Object Detection | [논문](https://arxiv.org/abs/1506.02640), 설명, 구현 |
- [7] YOLOv4: Optimal Speed and Accuracy of Object Detection | [논문](https://arxiv.org/abs/2004.10934), 설명, 구현 | 

---

## Quick paper reviews

### #1
#### Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

본 논문이 나오기 이전까지의 object detection 분야에서는 가장 높은 성능을 달성한 SOTA 모델로 SPP-Net과 Fast R-CNN 등이 있었다. 두 모델은 그 이전에 제안되었던 R-CNN보다 상대적으로 속도가 더욱 빨랐지만, 여전히 네트워크 바깥에서 CPU 방식으로 돌아가는 region proposal 단계에서 많은 시간이 소요된다는 단점이 존재했다.

본 논문에서는, 네트워크와 별개로 특정 영역을 CPU 단에서 추출했던 방식에서 벗어나 GPU에서 end-to-end로 학습이 가능한 Region Proposal Network(RPN)을 제안하여 region proposal에 필요한 시간을 획기적으로 단축하는 것은 물론, 성능 역시 향상시켰다. RPN은 어떠한 위치(bounding box)에 물체가 존재하는가(0/1)를 학습하는 모듈이다. 여기서 물체가 존재한다면(=1), 이 box 내 Object가 어떤 class인지 분류하는 것은 기존의 Fast R-CNN의 classifier 구조를 활용한다. 따라서 Faster R-CNN은 기존의 Fast R-CNN 모델 구조에 Region Propasal Network 모듈을 합친 형태인 것이다.

<p align="center">
<img width="476" alt="스크린샷 2022-02-18 오후 11 07 08" src="https://user-images.githubusercontent.com/63924704/154697545-eeab20f5-55e2-4081-9fa3-b0e7fc7cc0cb.png">
</p>

<p align="center">
<img width="1145" alt="스크린샷 2022-02-18 오후 11 12 26" src="https://user-images.githubusercontent.com/63924704/154698335-592a612f-6552-4026-821f-9adf4287220c.png">
</p>

보다 자세한 설명: [링크](https://seunghyun.oopy.io/4bb355c8-d44a-45bb-ac1f-11acffc69d60)

---