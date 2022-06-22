# ViT model implementation

import torch
import torch.nn as nn

# Linear Projection of Flattened Patches
# (1 x P^2C)크기의 하나의 이미지 패치를 (1 x D)로 임베딩
class LinearProjection(nn.Module):
    
    def __init__(self, patch_vec_size, num_patches, latent_vector_size):
        super().__init__()
        self.linear_projection = nn.Linear(patch_vec_size, latent_vector_size)
        self.class_token = nn.Parameter(torch.randn(1, latent_vector_size)) # 1xD 크기의 학습 가능한 파라미터를 concat해야 한다. 
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1, latent_vector_size)) # n_patches+class_token
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # input x의 크기: [Batch size x num_patches x (C x P x P)]
        batch_size = x.size(0)
        cls_token = self.class_token.repeat(batch_size, 1, 1) # [1xD] -> [B x 1 x D]
        
        x = self.linear_projection(x) # [B x num_patches x (C x P^2)] -> [B x num_patches x D]
        x = torch.cat([cls_token, x], dim=1) # [B x num_patches x D] -> [B x num_patches+1 x D]
        x += self.positional_embedding
        x = self.dropout(x)
        
        return x
        
        
# Multi-Head self Attention
class MSA(nn.Module):
    def __init__(self, latent_vector_size, num_heads):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.latent_vector_size = latent_vector_size
        self.head_dim = int(latent_vector_size / num_heads) # D_h = (D/k)
        
        self.Q = nn.Linear(latent_vector_size, latent_vector_size) # [D] -> [k*D_h(D_h=D/k)] : Multi head가 한번에 계산된다는 의미
        self.K = nn.Linear(latent_vector_size, latent_vector_size)
        self.V = nn.Linear(latent_vector_size, latent_vector_size)
        
        self.scale = torch.sqrt(latent_vector_size*torch.ones(1)).to(device) # 학습 파라미터가 아니기 때문에, cpu로 존재. gpu로 올려주자.
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        q = self.Q(x) # B x num_patches+1 x D
        k = self.K(x)
        v = self.V(x)
        
        # 원래는 [B x num_patches+1 x head_dim x num_heads] 의 4차원 tensor를 계산해야하는데, 3차원으로 계산하고 나중에 4차원으로 쪼개주자.
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3) # [B x num_heads x num_patches+1 x head_dim]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1) # q와 내적을 위해 Transpose해준다.
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        
#         print('q: ', q.shape)
#         print('k: ', k.shape)
#         print('v: ', v.shape)
        
        qk = torch.matmul(q, k) / self.scale
        
        A = torch.softmax(qk, dim=-1) # A=attention
        A = self.dropout(A)
        
        x = torch.matmul(A, v) # [B x num_heads x num_patches+1 x head_dim]
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.latent_vector_size) # [B x num_patches+1 x D] (Input과 동일한 크기로 맞춰준다)
        
        return x, A
        
        
class TransformerEncoder(nn.Module):
    '''
    The Transformer encoder consists of alternating layers of multiheaded self-attention and MLP blocks. 
    Layernorm is applied before every block, and residual connections after every block.
    '''
    def __init__(self, latent_vector_size, num_heads, hidden_dim):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(latent_vector_size)
        self.ln2 = nn.LayerNorm(latent_vector_size)
        self.msa = MSA(latent_vector_size, num_heads) # Multi-Head self Attention
        self.dropout = nn.Dropout(0.1)
        
        # The MLP contains two layers with a GELU non-linearity.
        self.mlp = nn.Sequential(
            nn.Linear(latent_vector_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_vector_size), # [B x num_patches+1 x D]
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        
        # x: Embedded Patches
        z = self.ln1(x) # 1. Layer Normalization
        z, attention = self.msa(z) # 2. Multi-Head Attention
        z = self.dropout(z) 
        z_p = z + x # 3. Residual Connection
        
        z = self.ln2(z_p) # 4. Layer Normalization
        z = self.mlp(z) # 5. MLP
        z += z_p # 5. Layer Normalization
        
        return z, attention
    
    
# ViT
class ViT(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vector_size, num_heads, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.linear_projection = LinearProjection(patch_vec_size, num_patches, latent_vector_size)
        
        self.transformer = nn.ModuleList()
        
        for _ in range(num_layers):
            self.transformer.append(TransformerEncoder(latent_vector_size, num_heads, hidden_dim))
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(latent_vector_size),
            nn.Linear(latent_vector_size, num_classes)
        )
        
    def forward(self, x):
        att_list = []
        x = self.linear_projection(x)
        for encoder in self.transformer:
            x, att = encoder(x)
            att_list.append(att)
            
        # [B x num_patches+1 x D] -> [B x D](가장 앞에 있는 class token만 떼어온다.)
        x = self.mlp_head(x[:, 0]) # Only class token
        
        return x, att  # [B x Num classes]
        

if __name__ == '__main__':
    
    # settings
    img_size = 32
    patch_size = 4

    num_patches = int((img_size/patch_size)**2)
    latent_vector_size = 128
    patch_vec_size = patch_size**2 * 3
    num_heads = 8
    hidden_dim = 128
    num_layers = 12
    num_classes = 10
    
    # ViT
    model = ViT(patch_vec_size, num_patches, latent_vector_size, num_heads, hidden_dim, num_layers, num_classes)
    
    # Sample input
    x = torch.rand((8, num_patches, patch_vec_size))
    
    output, att = model(x)
    print(output.shape)