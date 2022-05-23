## 논문 리스트

### Transformers
- [1] Attention Is All You Need | [논문](https://arxiv.org/abs/1706.03762), 설명, 구현 | 
- [2] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | [논문](https://arxiv.org/abs/2010.11929), 설명, 구현 |
- [3] How Do Vision Transformers Work? | [논문](https://arxiv.org/abs/2202.06709), 설명, 구현 | 
- [4] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | [논문](https://arxiv.org/abs/2103.14030), 설명, 구현 |

### Data Augmentation
- [1] mixup: Beyond Empirical Risk Minimization | [논문](https://arxiv.org/abs/1710.09412), [설명](#1), 구현 |
- [2] CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features | [논문](https://arxiv.org/abs/1905.04899), 설명, 구현 | 

### 3D
- [1] PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation | [논문](https://arxiv.org/abs/1612.00593), 설명, [구현](https://github.com/sseunghyuns/ai-paper-study/tree/main/Others/PointNet) |

### Ensembles
- [1] Weighted boxes fusion: Ensembling boxes from different object detection models | [논문](https://arxiv.org/abs/1910.13302), 설명, 구현 |

### Multi-modal
- [1] Show and Tell: A Neural Image Caption Generator | [논문](https://arxiv.org/abs/1411.4555), 설명, 구현 |  

### Pose Estimation
- [1] Stacked Hourglass Networks for Human Pose Estimation | [논문](https://arxiv.org/abs/1603.06937), 설명, 구현 |

### Meta-learning
- [1] Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks | [논문](https://arxiv.org/abs/1703.03400), 설명, 구현 |

### Optical Character Recognition
- [1] EAST: An Efficient and Accurate Scene Text Detector | [논문](https://arxiv.org/abs/1704.03155), 설명, 구현 |


---

## Quick paper reviews

### #1
#### mixup: Beyond Empirical Risk Minimization

* 기존 Empirical Risk Minimization(ERM) 방법으로 학습된 크고 깊은 모델들은 강력하지만 adversarial examples에 대해 memorization(과적합)과 sensitivity의 문제를 보인다. 이에 대해 mixup , 즉 convex combinations 방식의 데이터 증강 기법을 적용하여 모델의 과적합을 줄이고 예측 강건함(robustness)를 높인다.

---
