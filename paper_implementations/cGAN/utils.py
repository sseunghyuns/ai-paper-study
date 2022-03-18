import torch
import numpy as np
from torchvision.utils import save_image

def sample_image(n_row, epoch, generator):
    size = 28
    # Sample noise
    z = torch.randn(n_row, 100).type(torch.FloatTensor).cuda()
    gen_labels = []
    
    # Get labels ranging from 0 to n_classes for n rows
    for randpos in np.random.randint(0, 10, n_row):
      gen_labels.append(torch.eye(10)[randpos])
    gen_labels = torch.stack(gen_labels).cuda()
    generator.eval()
    gen_imgs = generator(z, gen_labels)
    save_image(gen_imgs.view(n_row,1,size,size).data, "images/%d.png" % epoch, nrow=n_row, normalize=True)
