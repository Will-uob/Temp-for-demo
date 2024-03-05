import os
import io
import sys
from typing import Union, Annotated
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.logger import logger
# from pydantic import BaseSettings

import torch
from diffusers import (StableDiffusionXLPipeline, DPMSolverMultistepScheduler,
                       StableDiffusionXLAdapterPipeline, T2IAdapter, 
                       EulerAncestralDiscreteScheduler, AutoencoderKL)
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.pidi import PidiNetDetector
from PIL import Image
from utils import pillow_image_to_base64_string

# Loading stuff for the application

app = FastAPI()

adapter = T2IAdapter.from_pretrained(
  "TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16, varient="fp16"
).to("cuda")

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    model_id, vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
).to("cuda")

pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/models/fillitin")
async def fillitin(prompt: str, negative_prompt: Union[str, None] = "lowres", resolution: Union[int, None] = 1024, 
                   sketch_guidance: Union[float, None] = 0.7, classifier_guidance: Union[float, None] = 7.5, submitfile: UploadFile = File(...)):
    if prompt is None or submitfile is None:
        return {'Status': 'Failure, prompt and submitfile not provided'}
    else:
        print("===== About the file ======")
        print(submitfile.filename)
        print(submitfile.content_type)
        print(submitfile)
        contents = "Hello"
        try:
            contents = submitfile.file.read()
        except Exception:
            return {"message": "There was an error uploading the file"}
    
        print(f"contents of the image: {contents}")
        # image = Image.open(io.BytesIO(contents))
        print(submitfile.file.read())
        print(submitfile.file)
        image = Image.open(io.BytesIO(contents))
        
        image = pidinet(  # Pixel difference image convolution
            image, detect_resolution=resolution, image_resolution=resolution, apply_filter=True
        )

        # negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"
        gen_images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=30,
            adapter_conditioning_scale=sketch_guidance,  # The outputs of the adapter are multiplied by `adapter_conditioning_scale` before they are added to the
                # residual in the original unet. If multiple adapters are specified in init, you can set the corresponding scale as a list.
            guidance_scale=classifier_guidance,  # Scale in accordance with classifier free guidance
            width=resolution,
            height=resolution
        ).images[0]
        dim = gen_images.size

        data_url = 'data:image/jpeg;base64,' + pillow_image_to_base64_string(gen_images)
        print(data_url)

    return {"dimensions": dim, "image_bytes": gen_images.tobytes("hex", "rgb"), "data_url": data_url}

@app.post("/uploadfile/")
def create_upload_file(q: str, file: UploadFile):
    return {"filename": file.filename, 'q': q}
