import torch
from vit import ViT

model = ViT(
    img_size=224,
    patch_size=16,
    in_ch=3,
    num_classes=1000,
    embed_dim=768,
    depth=2,
    num_heads=12,
    mlp_ratio=4,
    drop_rate=0.0,
)

# n of pathes 197
# 768 transformed size of each patch from 16*16
# attn shape = [1, 12, 197, 197]
# shape out from attention [1, 197, 768]
img = torch.randn(1, 3, 224, 224)
pred = model(img)
