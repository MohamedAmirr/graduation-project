# import base64
import json
import ast
import uuid

import openai
from PIL import Image, ImageDraw, ImageFont

# from io import BytesIO
# from django.contrib.auth.decorators import login_required
# from django.shortcuts import render
from diffusers import DiffusionPipeline, AutoencoderKL

import torch

# from PIL import Image
# import matplotlib.pyplot as plt
# from django.views.decorators.csrf import csrf_exempt
# from .forms import MyForm
# import os
from openai import OpenAI
import os

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch


from GP import settings
from home.forms import StoryForm
from home.helpers import check_words
from home.models import Scene, Story


def generate_story(keywords, num_scenes):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    example_json = {"title": "", "story": [{"paragraph": "", "prompt": ""}]}

    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a story writer, provide output in valid JSON. The home schema should be like this: "
                + json.dumps(example_json),
            },
            {
                "role": "user",
                "content": f"Create a story with {num_scenes} scenes using Disney characters based on this description: {keywords}",
            },
        ],
    )
    return json.loads(chat_completion.choices[0].message.content)


def generate_image(prompt):
    pass
    # Create an image with Pillow (replace this with your actual image generation logic)
    # img = Image.new("RGB", (300, 300), color=(73, 109, 137))
    # d = ImageDraw.Draw(img)
    # d.text((10, 10), prompt, fill=(255, 255, 0))
    #
    # # Save the image to the server
    # image_name = f"{uuid.uuid4()}.png"
    # image_path = os.path.join(settings.MEDIA_ROOT, image_name)
    # img.save(image_path)
    #
    # # Return the relative path to the image
    # return os.path.join(settings.MEDIA_URL, image_name)


def generate_story_view(request):
    if request.method == "POST":
        form = StoryForm(request.POST)
        if form.is_valid() and form.clean_description():
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


def generate_images_page(request):
    return render(request, "generate_images.html")


def save_image(image, prompt):
    # Create a directory for saving images
    media_dir = "media/generated_images"
    if not os.path.exists(media_dir):
        os.makedirs(media_dir)

    # Save the image
    image_name = f"{prompt[:50].replace(' ', '_')}.png"
    image_path = os.path.join(media_dir, image_name)
    image.save(image_path)

    return image_path


@csrf_exempt
def generate_images(request):
    if request.method == "POST":
        scenes = request.POST.getlist("scenes[]")

        # Check for abusive content
        if any(check_words(scene) == "yes" for scene in scenes):
            return JsonResponse({"error": "Abusive content detected"}, status=400)

        # Generate prompts using OpenAI API
        prompts = []
        for scene in scenes:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            example_json = {"title": "", "story": [{"paragraph": "", "prompt": ""}]}

            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a story writer, provide output in valid JSON. The home schema should be like this: "
                        + json.dumps(example_json),
                    },
                    {
                        "role": "user",
                        "content": f"Create a prompt for the scene: {scene}",
                    },
                ],
            )
            response = chat_completion.choices[0].message.content
            prompt_json = json.loads(response)
            prompts.append(prompt_json["story"][0]["prompt"])
            
        pipe = StableDiffusionXLPipeline.from_single_file(
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors", 
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")

        # Generate images using the provided code
        # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
        # pipe = DiffusionPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-base-1.0",
        #     vae=vae,
        #     torch_dtype=torch.float16,
        #     use_safetensors=True,
        # )

        image_urls = []
        for prompt in prompts:
            image = pipe(prompt, num_inference_steps=1).images[0]
            image_path = save_image(image, prompt)
            image_urls.append(image_path)
            

        # Save story data in session for display
        request.session["story_data"] = {
            "title": "Generated Story",
            "story": [
                {"paragraph": scene, "image_url": image_url}
                for scene, image_url in zip(scenes, image_urls)
            ],
        }

        return JsonResponse({"images": image_urls})


# @csrf_exempt
# def generate_images(request):
#     if request.method == "POST":
#         form = MyForm(request.POST)
#         if form.is_valid():
#             vae = AutoencoderKL.from_pretrained(
#                 "madebyollin/sdxl-vae-fp16-fix",
#                 # torch_dtype=torch.float32
#             )
#
#             pipe = DiffusionPipeline.from_pretrained(
#                 "stabilityai/stable-diffusion-xl-base-1.0",
#                 # vae=vae,
#                 # torch_dtype=torch.float16,
#                 # variant="fp16",
#                 # use_safetensors=True,
#             )
#             # pipe = pipe.to("cuda")
#             prompt = form.cleaned_data["story"]
#
#             # images = pipe(prompt=prompt, num_inference_steps=25, num_images_per_prompt=1)
#             # image = images.images[0]
#             img_str = []
#             image = Image.new("RGB", (200, 100), color="red")
#             draw = ImageDraw.Draw(image)
#             draw.text((10, 10), "Hello", fill="white")
#             buffered = BytesIO()
#             image.save(buffered, format="PNG")
#             img_str.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
#
#             image = Image.new("RGB", (200, 100), color="blue")
#             draw = ImageDraw.Draw(image)
#             draw.text((10, 10), "Hello", fill="white")
#             buffered = BytesIO()
#             image.save(buffered, format="PNG")
#             img_str.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
#
#             # image_grid(image.images, 1, 1)
#             return render(request, "index.html", {"generated_images": img_str})
#     else:
#         form = MyForm()
#
#     return render(request, "index.html", {"form": form})


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
        story = Story.objects.create(title=story_data["title"], description=description)
        story.users.add(request.user)

        # Create and save the scenes
        for scene_data in story_data.get("story", []):
            sentence = scene_data.get("paragraph", "")
            image_url = scene_data.get("image_url", "")
            Scene.objects.create(story=story, sentence=sentence, image=image_url)
        is_saved = True
        return render(
            request,
            "display_story.html",
            {
                "story_data": story_data,
                "story": json.dumps(story_data),
                "is_saved": is_saved,
            },
        )

    return render(
        request,
        "display_story.html",
        {"story_data": story_data, "story": json.dumps(story_data)},
    )


User = get_user_model()


@csrf_exempt
def check_user_email(request):
    if request.method == "POST":
        data = json.loads(request.body)
        email = data.get("email")
        story_data = data.get("story_data")

        user_exists = User.objects.filter(email=email).exists()

        if user_exists:
            user = User.objects.get(email=email)
            # Create a new story for the target user
            story = Story.objects.create(title="Shared Story")
            story.users.add(user)
            # Add scenes to the story
            for scene_data in story_data["story"]:
                Scene.objects.create(
                    story=story,
                    sentence=scene_data["paragraph"],
                    image=scene_data["image_url"],
                )
            return JsonResponse({"exists": True})
        return JsonResponse({"exists": False})
    return JsonResponse({"error": "Invalid request method."}, status=400)


@login_required
def user_stories_view(request):
    stories = request.user.stories.all()
    return render(request, "user_stories.html", {"stories": stories})


@login_required
def story_detail_view(request, story_id):
    story = get_object_or_404(Story, id=story_id, users=request.user)
    scenes = story.scenes.all()
    story_data = {
        "title": story.title,
        "story": [
            {"image_url": scene.image, "paragraph": scene.sentence} for scene in scenes
        ],
    }
    is_saved = story.id is not None
    return render(
        request,
        "display_story.html",
        {
            "story_data": story_data,
            "story": str(json.dumps(story_data)),
            "is_saved": is_saved,
        },
    )
