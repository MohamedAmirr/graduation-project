from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
    DiffusionPipeline,
)
from diffusers.models.autoencoders.vq_model import VQEncoderOutput
from diffusers.models.autoencoders.vq_model import VQModel
import torch

# Load the VAE model
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")

# Load the pipeline model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe = pipe.to("cuda")


# Example usage to generate an image
def generate_image(prompt):
    return pipe(prompt, num_inference_steps=20).images[
        0
    ]  # Reduce the number of steps for faster generation


# Test the function
if __name__ == "__main__":
    prompt = "A fantasy landscape with mountains and a river"
    image = generate_image(prompt)
    image.save("output.png")
