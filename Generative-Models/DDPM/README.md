**Denoising Diffusion Probabilistic Models**

</br>

$\overbrace{\text{Image}(x_0) \text{ → noise → noise}(x_t) \ \text{→ … → noise}(x_T)}^{\color{green}{\text{Forward Process(Non parameterized)}}}$

$\underbrace{\text{Image}(x_0) \text{ ← noise ← noise}(x_t) \text{ ← … ← noise}(x_T)}_{\color{violet}\text{Reverse process(Parameterized)}}$

Diffusion 모델도 VAE, GAN 등과 같이 latent variable $z$를 조건부로 하여 true data distribution인 $q(x_0)$를 근사하는 $p_{\theta}(x|z)$를 학습시켜 새로운 데이터를 sampling하는 모델이다. 이를 위해 먼저 input data $x_0$을 $z$로 mapping하는 **forward process**를 거치게 된다. VAE에서 encoder에 해당하는 부분이다. 하지만 VAE와는 다르게, Diffusion의 forward process는 학습 대상이 아니다. 사전에 정의된 schedule에 따라 총 $T$ 스텝에 걸쳐 $x_0$에 gaussian noise를 조금씩 더하여, 최종적으로 $x_T$를 완전한 gaussian noise가 되게끔 만드는 것이다. 그 후, 다시 $x_T$에서 시작하여 Markov chain으로 정의된 $T$번의 **reverse process** $p_{\theta}(x_{t-1}|x_t)$를 거친 $p_{\theta}(x|z)$가 true data distribution $q(x_0)$와 비슷해지도록 학습을 진행한다. 

여기서, forward process $q(x_{t}|{x_t-1})$의 조건부를 바꾼 ${q(x_{t-1}|{x_t})}$를 구할 수 있다면 모델 학습이 필요 없을 것이다. 하지만 이를 직접적으로 계산할 수 없기 때문에 $p_{\theta}(x_{t-1}|x)$를 학습하여  $q(x_{t-1}|x_t)$를 근사하고자 하는 것이다. 또한 주입되는 gaussian noise의 variance인 $\beta_t$가 매우 작을 때 $q(x_{t}|x_{t-1})$와 $q(x_{t-1}|x_t)$ 모두 gaussian 분포를 따른다는 것은 1949년의 한 논문에서 이미 증명되었고, 이를 바탕으로 Diffusion도 forward와 reverse process 모두 조건부 gaussian 분포로 놓는 학습 전략을 세우게 된다.

</br>

<img src="figure/movie.gif" width="250" align="center">

</br>

- Trained CelebA dataset for 20 epochs. Definitely need to train more to get better results.
