import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
# setup device to use
# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
device = "cuda:3"
# load sample image
raw_image = Image.open("/mnt/d/SPD_2025/data/UCY/students03/bg.png").convert("RGB")
# display(raw_image.resize((596, 437)))
import torch

# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
# # prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
#给定图像和文本提示，让模型生成响应。
model.generate({"image": image, "prompt": "What objects are there in the picture? Answer:"})

# model.generate({
#     "image": image,
#     "prompt": "Question: which city is this? Answer: singapore. Question: why?"})