import base64
import json
from io import BytesIO
from PIL import ImageDraw, Image
from django.shortcuts import render
from diffusers import DiffusionPipeline, AutoencoderKL
import torch
from PIL import Image
import matplotlib.pyplot as plt
from django.views.decorators.csrf import csrf_exempt
from .forms import MyForm
import os
from openai import OpenAI


def generate_story(keywords):
    client = OpenAI(
        api_key='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'
    )
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a skilled in composing cartoon stories."},
            {"role": "user",
             "content": keywords}
        ],
    )
    print(chat_completion.choices[0].message.content)


def image_grid(imgs, rows, cols, resize=256):
    assert len(imgs) == rows * cols

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]

    plt.figure(figsize=(20, 20))

    for index, image in enumerate(imgs):
        ax = plt.subplot(1, len(imgs), index + 1)
        plt.imshow(image)
        plt.axis("off")
    plt.show()


@csrf_exempt
def generate_images(request):
    if request.method == "POST":
        form = MyForm(request.POST)
        if form.is_valid():
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                # torch_dtype=torch.float32
            )

            pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                # vae=vae,
                # torch_dtype=torch.float16,
                # variant="fp16",
                # use_safetensors=True,
            )
            # pipe = pipe.to("cuda")
            prompt = form.cleaned_data['story']

            # images = pipe(prompt=prompt, num_inference_steps=25, num_images_per_prompt=1)
            # image = images.images[0]
            image = Image.new('RGB', (200, 100), color='blue')
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), "Hello", fill="white")
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # image_grid(image.images, 1, 1)
            return render(request, 'index.html', {'generated_images': img_str})
    else:
        form = MyForm()

    return render(request, 'index.html', {'form': form})
