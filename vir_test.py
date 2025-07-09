import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
virchow = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
config = resolve_data_config(virchow.pretrained_cfg, model=virchow)
transform = create_transform(**config)

print("pre_config", virchow.pretrained_cfg)
print("config",config)
print("transform", transform)
