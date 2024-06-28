# import base64
import json

# from io import BytesIO
# from PIL import ImageDraw, Image
# from django.contrib.auth.decorators import login_required
# from django.shortcuts import render
# from diffusers import DiffusionPipeline, AutoencoderKL
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# from django.views.decorators.csrf import csrf_exempt
# from .forms import MyForm
# import os
# from openai import OpenAI
import os

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt

from home.forms import StoryForm
from home.models import Scene, Story


def generate_story(keywords, num_scenes):
    return {
        "story": [
            {
                "paragraph": f"This is a paragraph for scene 1 based on {keywords}.",
                "prompt": "Disney character scene 1",
            },
            {
                "paragraph": f"This is a paragraph for scene 2 based on {keywords}.",
                "prompt": "Disney character scene 2",
            },
        ][:num_scenes]
    }


#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     example_json = {"story": [{"paragraph": "", "prompt": ""}]}
#
#     chat_completion = client.chat.completions.create(
#         model="gpt-3.5-turbo-1106",
#         response_format={"type": "json_object"},
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Provide output in valid JSON. The home schema should be like this: "
#                 + json.dumps(example_json),
#             },
#             {
#                 "role": "user",
#                 "content": f"Create a short story with {num_scenes} scenes using Disney characters based on the following description: {keywords}",
#             },
#         ],
#     )
#     return json.loads(chat_completion.choices[0].message.content)


def generate_image(prompt):
    # Implement your image generation logic here
    # For now, let's return a placeholder URL
    return "https://via.placeholder.com/150"


def generate_story_view(request):
    if request.method == "POST":
        form = StoryForm(request.POST)
        if form.is_valid():
            description = form.cleaned_data["description"]
            num_scenes = form.cleaned_data["num_scenes"]
            story_data = generate_story(description, num_scenes)

            # Generate images for each scene
            for scene in story_data["story"]:
                scene["image_url"] = generate_image(scene["prompt"])

            # Save story data to session
            request.session["story_data"] = story_data
            request.session["description"] = description

            return redirect("display_story")
    else:
        form = StoryForm()

    return render(request, "generate_story.html", {"form": form})


def image_grid(imgs, rows, cols, resize=256):
    pass
    # assert len(imgs) == rows * cols
    #
    # if resize is not None:
    #     imgs = [img.resize((resize, resize)) for img in imgs]
    #
    # plt.figure(figsize=(20, 20))
    #
    # for index, image in enumerate(imgs):
    #     ax = plt.subplot(1, len(imgs), index + 1)
    #     plt.imshow(image)
    #     plt.axis("off")
    # plt.show()


@csrf_exempt
def generate_images(request):
    pass
    # if request.method == "POST":
    #     form = MyForm(request.POST)
    #     if form.is_valid():
    #         vae = AutoencoderKL.from_pretrained(
    #             "madebyollin/sdxl-vae-fp16-fix",
    #             # torch_dtype=torch.float32
    #         )
    #
    #         pipe = DiffusionPipeline.from_pretrained(
    #             "stabilityai/stable-diffusion-xl-base-1.0",
    #             # vae=vae,
    #             # torch_dtype=torch.float16,
    #             # variant="fp16",
    #             # use_safetensors=True,
    #         )
    #         # pipe = pipe.to("cuda")
    #         prompt = form.cleaned_data["story"]
    #
    #         # images = pipe(prompt=prompt, num_inference_steps=25, num_images_per_prompt=1)
    #         # image = images.images[0]
    #         img_str = []
    #         image = Image.new("RGB", (200, 100), color="red")
    #         draw = ImageDraw.Draw(image)
    #         draw.text((10, 10), "Hello", fill="white")
    #         buffered = BytesIO()
    #         image.save(buffered, format="PNG")
    #         img_str.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    #
    #         image = Image.new("RGB", (200, 100), color="blue")
    #         draw = ImageDraw.Draw(image)
    #         draw.text((10, 10), "Hello", fill="white")
    #         buffered = BytesIO()
    #         image.save(buffered, format="PNG")
    #         img_str.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    #
    #         # image_grid(image.images, 1, 1)
    #         return render(request, "index.html", {"generated_images": img_str})
    # else:
    #     form = MyForm()
    #
    # return render(request, "index.html", {"form": form})


def home(request):
    return render(request, "home/home.html")


def display_story_view(request):
    story_data = request.session.get("story_data")
    description = request.session.get("description")

    if (
        request.method == "POST"
        and request.user.is_authenticated
        and "save" in request.POST
    ):
        # Create and save the story for authenticated users
        story = Story.objects.create(title="Generated Story", description=description)
        story.users.add(request.user)

        # Create and save the scenes
        for scene_data in story_data.get("story", []):
            sentence = scene_data.get("paragraph", "")
            image_url = scene_data.get("image_url", "")
            Scene.objects.create(story=story, sentence=sentence, image=image_url)

        return redirect("home")  # Redirect to home or any other page after saving

    return render(request, "display_story.html", {"story_data": story_data})
