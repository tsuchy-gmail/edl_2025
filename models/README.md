---
license: apache-2.0
tags:
- image-feature-extraction
- PyTorch
- timm
extra_gated_prompt: >

  The Virchow Model and associated code are released under the Apache License, Version 2.0 (the "License"). You may obtain a copy of the License at:
  
  http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
  
  Additional Terms

  Please note that the primary email used to sign up for your Hugging Face account must match your institutional email to receive approval. By downloading the Virchow Model, you attest that all information (affiliation, research use) is correct and up-to-date. Downloading the Virchow Model requires prior registration on Hugging Face and agreeing to the terms of use.
  
  While the Apache 2.0 License grants broad permissions, we kindly request that users adhere to the following guidelines:
  
    1. Attribution: We encourage proper attribution when using or redistributing the Virchow Model or its derivatives. Please include a reference to the original source and creators.
    
    2. Responsible Use: Users are expected to use the Virchow Model responsibly and ethically. Please consider the potential impacts of your use on individuals and society.
    
    3. Medical or Clinical Use: The Virchow Model is not intended for use in medical diagnosis, treatment, or prevention of disease of real patients. It should not be used as a substitute for professional medical advice.
    
    4. Privacy and Data Protection: Users should respect privacy rights and comply with applicable data protection laws when using the Virchow Model.
    
    5. No Malicious Use: The Virchow Model should not be used to create malicious code, malware, or to interfere with the proper functioning of computer systems.
    
    6. Transparency: If you use the Virchow Model in a product or service, we encourage you to disclose this fact to your end users.
    
    7. Feedback and Contributions: We welcome feedback and contributions to improve the Virchow Model. Please consider sharing your improvements with the community.
    
  These additional terms are not intended to restrict your rights under the Apache 2.0 License but to promote responsible and ethical use of the Virchow Model.
  
  By using the Virchow Model, you acknowledge that you have read and understood these terms.
extra_gated_fields:
  First and Last Name: text
  Institutional Email (must match your primary HuggingFace email): text
  I agree to the license and terms of use described above: checkbox
---

> [!IMPORTANT]
> 🎉 Check out the latest version of Virchow here: [Virchow2](https://huggingface.co/paige-ai/Virchow2)

# Model card for Virchow

Virchow is a self-supervised vision transformer pretrained using 1.5M whole slide histopathology images. The model can be used as a tile-level feature extractor (frozen or finetuned) to achieve state-of-the-art results for a wide variety of downstream computational pathology use cases.

## Model Details
- **Developed by:** Paige, NYC, USA and Microsoft Research, Cambridge, MA USA
- **Model Type:** Image feature backbone
- **Model Stats:**
  - Params (M): 632
  - Image size: 224 x 224
- **Model Architecture:**
  - Architecture: ViT-H/14
  - Patch size: 14
  - Layers: 32
  - Embedding dimension: 1280
  - Activation function: SwiGLU
  - Attention heads: 16
  - LayerScale: true
- **Training Details:**:
  - Precision: Mixed precision (`fp16`)
  - Objective: DINOv2 (https://doi.org/10.48550/arXiv.2304.07193)
- **Paper:**
  - A foundation model for clinical-grade computational pathology and rare cancers detection: https://www.nature.com/articles/s41591-024-03141-0
- **Pretraining Dataset:** Internal dataset of 1.5 million whole slide images from Memorial Sloan Kettering Cancer Center, tiles sampled at 0.5 microns per pixel resolution (20x magnification).
- **License:** Apache 2.0

## Model Usage

### Requirements
  - PyTorch (2.0+ recommended)
  - timm (>= 0.9.11 required)
  - huggingface_hub

### Login
After gaining access to the model here, you will need to login to HuggingFace in the environment you wish to use the model. This can be done from the command line:

```
huggingface-cli login
```

or in your Python code:

```python
from huggingface_hub import login

login()
```

Please refer to official HuggingFace [documentation](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) for more details.

### Image Embeddings
```python
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image

# need to specify MLP layer and activation function for proper init
model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
model = model.eval()

transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

image = Image.open("/path/to/your/image.png")
image = transforms(image).unsqueeze(0)  # size: 1 x 3 x 224 x 224

output = model(image)  # size: 1 x 257 x 1280

class_token = output[:, 0]    # size: 1 x 1280
patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280

# concatenate class token and average pool of patch tokens
embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
```
We concatenate the class token and the mean patch token to create the final tile embedding. In more resource constrained settings, one can experiment with using just class token or the mean patch token. For downstream tasks with dense outputs (i.e. segmentation), the `256 x 1280` tensor of patch tokens can be used.

We highly recommend running the model on a GPU in mixed precision (`fp16`) using `torch.autocast`:

```python
model = model.to("cuda")
image = image.to("cuda")

with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
  output = model(image)

class_token = output[:, 0]
patch_tokens = output[:, 1:]

embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)

# the model output will be fp32 because the final operation is a LayerNorm that is ran in mixed precision
# optionally, you can convert the embedding to fp16 for efficiency in downstream use
embedding = embedding.to(torch.float16)
```

## Use

### Direct use
Virchow intended to be used as a frozen feature extractor as the foundation for tile-level and whole slide-level classifiers.

### Downstream use
Virchow can be finetuned to adapt to specific tasks and/or datasets.

## Terms of use

The Virchow Model and associated code are released under the Apache License, Version 2.0 (the "License"). You may obtain a copy of the License at:
  
http://www.apache.org/licenses/LICENSE-2.0
  
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
  
Additional Terms

Please note that the primary email used to sign up for your Hugging Face account must match your institutional email to receive approval. By downloading the Virchow Model, you attest that all information (affiliation, research use) is correct and up-to-date. Downloading the Virchow Model requires prior registration on Hugging Face and agreeing to the terms of use.
  
While the Apache 2.0 License grants broad permissions, we kindly request that users adhere to the following guidelines:
  
  1. Attribution: We encourage proper attribution when using or redistributing the Virchow Model or its derivatives. Please include a reference to the original source and creators.
    
  2. Responsible Use: Users are expected to use the Virchow Model responsibly and ethically. Please consider the potential impacts of your use on individuals and society.
    
  3. Medical or Clinical Use: The Virchow Model is not intended for use in medical diagnosis, treatment, or prevention of disease of real patients. It should not be used as a substitute for professional medical advice.
    
  4. Privacy and Data Protection: Users should respect privacy rights and comply with applicable data protection laws when using the Virchow Model.
    
  5. No Malicious Use: The Virchow Model should not be used to create malicious code, malware, or to interfere with the proper functioning of computer systems.
    
  6. Transparency: If you use the Virchow Model in a product or service, we encourage you to disclose this fact to your end users.
    
  7. Feedback and Contributions: We welcome feedback and contributions to improve the Virchow Model. Please consider sharing your improvements with the community.
    
These additional terms are not intended to restrict your rights under the Apache 2.0 License but to promote responsible and ethical use of the Virchow Model.
  
By using the Virchow Model, you acknowledge that you have read and understood these terms.

## Citation
Please cite the following work if you used this model in your research.

Vorontsov, E., Bozkurt, A., Casson, A. et al. A foundation model for clinical-grade computational pathology and rare cancers detection. Nat Med (2024). https://doi.org/10.1038/s41591-024-03141-0

```
@article{vorontsov2024virchow,
  title={A foundation model for clinical-grade computational pathology and rare cancers detection},
  author={Vorontsov, Eugene and Bozkurt, Alican and Casson, Adam and Shaikovski, George and Zelechowski, Michal and Severson, Kristen and Zimmermann, Eric and Hall, James and Tenenholtz, Neil and Fusi, Nicolo and Yang, Ellen and Mathieu, Philippe and van Eck, Alexander and Lee, Donghun and Viret, Julian and Robert, Eric and Wang, Yi Kan and Kunz, Jeremy D. and Lee, Matthew C. H. and Bernhard, Jan H. and Godrich, Ran A. and Oakley, Gerard and Millar, Ewan and Hanna, Matthew and Wen, Hannah and Retamero, Juan A. and Moye, William A. and Yousfi, Razik and Kanan, Christopher and Klimstra, David S. and Rothrock, Brandon and Liu, Siqi and Fuchs, Thomas J.},
  journal={Nature Medicine},
  year={2024},
  publisher={Nature Publishing Group}
}
```