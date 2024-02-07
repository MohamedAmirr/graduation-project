import base64
from io import BytesIO
from PIL import ImageDraw
from django.shortcuts import render
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
from .forms import MyForm


def create_text_image():
    image = Image.new('RGB', (200, 100), color='blue')
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "Hello", fill="white")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


@csrf_exempt
def generate_images(request):
    if request.method == "POST":
        form = MyForm(request.POST)
        if form.is_valid():
            image_list = []
            image_list.append(create_text_image())
            image_list.append(create_text_image())

            return render(request, 'index.html', {'generated_images': image_list})
    else:
        form = MyForm()

    return render(request, 'index.html', {'form': form})
