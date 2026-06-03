from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def main():
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)

    prompt = input("Enter image description: ").strip()
    if not prompt:
        print("Prompt cannot be empty.")
        return

    if not torch.cuda.is_available():
        print("CUDA is not available. Please run this on a system with an NVIDIA GPU.")
        return

    print("Loading model... This may take some time on first run.")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to("cuda")

    print("Generating image...")
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]

    output_path = result_dir / "output_image.png"
    image.save(output_path)

    print(f"Image saved successfully at: {output_path}")


if __name__ == "__main__":
    main()
