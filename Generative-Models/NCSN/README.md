**Generative Modeling by Estimating Gradients of the Data Distribution implementation using PyTorch**

</br>

This paper proposed a generative model that estimates new samples using the gradients of $p_{data}(x)$ estimated by score-matching. When the data exists in a low-dimensional manifold, the problem of inaccurate gradient estimation is solved by data perturbation with multiple levels of Gaussian. In addition, for sampling new data, the authors proposed annealed Langevin dynamics that use gradients with small noise as it get closed to the data manifold.

</br>

<img src="figure/movie.gif" width="250" align="center">

</br>

- Trained CelebA dataset for 6500 iterations. Definitely need to train more to get better results.
