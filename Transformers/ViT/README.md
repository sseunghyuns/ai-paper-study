### Vision Transformers implementation using PyTorch

</br>

**Set up**
- Trained CIFAR10(32x32)
- Patch size: 4 (A total of 64 patches are made)
- Embedding dimension(D): 128
- Number of heads: 8
- Number of layers: 12
- Number of MLP hidden dimension: 64

---

**Generate image patches**

- This is an example of patch generation. 
- width, height = 512, 384
- patch size = 32
- A total of 192(16*12) patch inputs are generated.
- For CIFAR10 data, there would be a smaller number of patches.

![patch_img](https://user-images.githubusercontent.com/63924704/174772049-17e3d936-0736-41cb-a1a6-fc6faf5cd4fc.jpg)

---

**ViT implementation in model.py**

<img width="800" alt="vit" src="https://user-images.githubusercontent.com/63924704/175024778-b98af135-5a90-49d3-88a8-9ffdfa89220f.png">
