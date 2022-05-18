

## 논문 리스트

---

### Generative Models

- [1] Generative Adversarial Networks | [논문](https://arxiv.org/abs/1406.2661), 설명, 구현 |
- [2] Conditional Generative Adversarial Nets | [논문](https://arxiv.org/abs/1411.1784), [설명](#1), [구현](https://github.com/sseunghyuns/ai-paper-study/tree/main/paper_implementations/cGAN) | 
- [3] Image-to-Image Translation with Conditional Adversarial Networks | [논문](https://arxiv.org/abs/1611.07004), [설명](#2), [구현](https://github.com/sseunghyuns/ai-paper-study/tree/main/paper_implementations/Pix2Pix) | 
- [4] Auto-Encoding Variational Bayes | [논문](https://arxiv.org/abs/1312.6114), 설명, 구현 | 
- [5] Denoising Diffusion Probabilistic Models | [논문](https://arxiv.org/abs/2006.11239), 설명, 구현 |
- [6] Generative Modeling by Estimating Gradients of the Data Distribution | [논문](https://arxiv.org/abs/1907.05600), 설명, 구현 |

---

## Quick paper reviews

### #1
#### Conditional Generative Adversarial Nets

기존 GAN 구조에 조건을 주어 모델의 데이터 생성을 특정 방향으로 유도할 수 있다는 것이 본 논문의 핵심이다. 이때 조건으로 사용되는 추가적인 정보는 class labels, 데이터의 특정 부분 inpainting, 다른 modality의 데이터 등 여러가지 형태로 존재할 수 있다. 

GAN을 우선 살펴보면, 두 개의 적대적 모델로 이루어져 있다. 
- A generative model **G**: 데이터의 분포를 capture
- A discriminator model **D**: G에 의해 생성된 데이터가 training set에서 온 것인지 판단

G와 D는 아래의 목적식을 최소화하는 방향으로 동시에 학습이 진행된다. 

- Generator(G): <img src="https://latex.codecogs.com/svg.image?log(1-D(G(z))" title="https://latex.codecogs.com/svg.image?log(1-D(G(z))" />
- Discriminator(D): <img src="https://latex.codecogs.com/svg.image?\bg{white}log(D(x))" title="https://latex.codecogs.com/svg.image?\bg{white}log(D(x))" />

이때 추가적인 정보 <img src="https://latex.codecogs.com/svg.image?\bg{white}y" title="https://latex.codecogs.com/svg.image?\bg{white}y" />가 주어지면 conditional model로 확장될 수 있다. <img src="https://latex.codecogs.com/svg.image?\bg{white}y" title="https://latex.codecogs.com/svg.image?\bg{white}y" />는 class labels 혹은 다른 modality의 데이터 등 어떠한 형태로든 존재할 수 있다.  이러한 <img src="https://latex.codecogs.com/svg.image?\bg{white}y" title="https://latex.codecogs.com/svg.image?\bg{white}y" />를 D와 G 에게 feeding함으로써 condition을 주는 것이다. 

<p align="center">
<img width="500" src="https://user-images.githubusercontent.com/63924704/158572477-63fa3af6-06f0-4df0-b8ab-2fb53b035b8f.png">
</p>

---

### #2
#### Image-to-Image Translation with Conditional Adversarial Networks

<p align="center">
<img width="800" src="https://user-images.githubusercontent.com/63924704/164194116-1e7afd20-91fb-4e2d-a3cd-a587a7de0d16.png">
</p>

Image-to-Image translation 적용 사례들에서 각각의 task별 이미지의 특성이 다르지만, 본 논문에서는 Pix2Pix 모델 구조의 동일한 방법을 사용하여 유의미한 결과를 도출했다. 지금까지 제안되었던 방법들은 각각의 task에 맞는 loss나 모델 구조를 제안했다면, 본 논문에서는 다양한 task에 일반화할 수 있는 방법을 제안한 것이다.

보다 자세한 설명: [링크](https://seunghyun.oopy.io/2ac5e525-90d8-4083-bc9a-b913ad3db1f4)


---
