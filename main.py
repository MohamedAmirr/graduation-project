import json

from diffusers import DiffusionPipeline, AutoencoderKL
import torch
from PIL import Image
import matplotlib.pyplot as plt
import intel_extension_for_pytorch

# vae = AutoencoderKL.from_pretrained(
#     "madebyollin/sdxl-vae-fp16-fix",
#     # torch_dtype=torch.float32
# )
#
# pipe = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     # vae=vae,
#     torch_dtype=torch.float16,
#     variant="fp16",
#     # use_safetensors=True,
# )
# pipe = pipe.to("cuda")
#
# prompt = "A man in a spacesuit is running a marathon in the jungle."
#
# image = pipe(prompt=prompt, num_inference_steps=25, num_images_per_prompt=1, height=10, width=10)
#
#
# def image_grid(imgs, rows, cols, resize=256):
#     assert len(imgs) == rows * cols
#
#     if resize is not None:
#         imgs = [img.resize((resize, resize)) for img in imgs]
#
#     plt.figure(figsize=(20, 20))
#
#     for index, image in enumerate(imgs):
#         ax = plt.subplot(1, len(imgs), index + 1)
#         plt.imshow(image)
#         plt.axis("off")
#     plt.show()

# w, h = imgs[0].size
# grid_w, grid_h = cols * w, rows * h
# grid = Image.new("RGB", size=(grid_w, grid_h))
#
# for i, img in enumerate(imgs):
#     x = i % cols * w
#     y = i // cols * h
#     grid.paste(img, box=(x, y))
#
# return grid


# image_grid(image.images, 1, 1)

from openai import OpenAI

client = OpenAI(api_key="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")

example_json = {"home": [{"paragraph": "", "prompt": ""}]}

chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "Provide output in valid JSON. The home schema should be like this: "
            + json.dumps(example_json),
        },
        {
            "role": "user",
            "content": "create a short home of six scenes and prompt to generate image using Disney characters",
        },
    ],
)
print(chat_completion.choices[0].message.content)
