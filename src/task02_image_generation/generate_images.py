import torch
from transformers import StableDiffusionPipeline
import os

class ImageGenerator:
    def __init__(self, model_name='CompVis/stable-diffusion-v1-4', output_dir='./generated_images'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Load the Stable Diffusion model pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_name).to(self.device)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_images(self, prompts, num_images=1, guidance_scale=7.5, height=512, width=512):
        """Generate images from prompts."""
        if isinstance(prompts, str):
            prompts = [prompts]

        generated_files = []
        for idx, prompt in enumerate(prompts):
            for i in range(num_images):
                image = self.pipeline(
                    prompt=prompt, 
                    guidance_scale=guidance_scale, 
                    height=height, 
                    width=width
                ).images[0]

                filename = os.path.join(self.output_dir, f"image_{idx + 1}_{i + 1}.png")
                image.save(filename)
                generated_files.append(filename)

        return generated_files
