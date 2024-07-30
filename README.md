# Story Image Generator "IMAGITALE"

## Project Overview:

  IMAGITALE is designed to empower users to create and generate their own unique Disney-style stories and images. The system utilizes advanced AI models to transform written narratives into vivid images, providing a seamless and interactive storytelling experience.


## Technologies Used:

1. OpenAI GPT-3.5: This powerful language model is used to generate detailed and coherent stories from user inputs.
2. Stable Diffusion XL (SDXL): A state-of-the-art image generation model that creates high-resolution, realistic images based on the story descriptions provided.
3. DreamBooth and LoRA: Techniques used for fine-tuning and improving the consistency and quality of generated images.


## Website Features:
1. User-Friendly Interface: Easy navigation and intuitive design to ensure accessibility for users of all ages and technical backgrounds.
2. Creative Empowerment: Tools for users to craft personalized stories with visual elements.
3. Educational Value: Helps in developing storytelling skills and creativity, especially among children.
4. Story and Image Generation: Allows users to input story descriptions and generate corresponding images, with options to add multiple scenes.


## Challenges and Solutions:
  One significant challenge we faced was ensuring consistency in character representation across different scenes. To address this:
      1. Fine-Tuning with DreamBooth and LoRA: Enhanced the model to specialize in rendering specific subjects consistently.
      2. Metadata Storage: Used metadata to maintain consistency in the appearance of characters and scenes.
      3. Batch Generation: Improved the efficiency and consistency of image generation by optimizing GPU memory usage.
