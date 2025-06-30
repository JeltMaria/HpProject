from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response
import numpy as np
import cv2
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import random
import io
import gc
import os
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()
executor = ThreadPoolExecutor(max_workers=1)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def resize_image(image: np.ndarray, max_size=512) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def extract_features(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        raise ValueError("Лица не обнаружены")
    x, y, w, h = faces[0]
    face_ratio = h / w
    face_shape = "oval" if face_ratio > 1.3 else "round"
    return {
        "face_shape": face_shape,
        "eye_color": random.choice(["blue", "brown", "green"]),
        "hair_color": random.choice(["black", "brown", "blonde"])
    }

def morph_features(features1, features2):
    return {
        "face_shape": random.choice([features1["face_shape"], features2["face_shape"]]),
        "eye_color": random.choice([features1["eye_color"], features2["eye_color"]]),
        "hair_color": random.choice([features1["hair_color"], features2["hair_color"]])
    }

def create_prompt(features, age_category):
    age_descriptions = {
        "toddler": "toddler (age 0-4, soft facial features, chubby cheeks, big eyes)",
        "child": "child (age 5-12, youthful appearance, smooth skin, bright expression)",
        "teen": "teenager (age 13-18, defined jawline, youthful features, clear skin)",
        "adult": "adult (over 18, mature facial features, defined cheekbones, natural look)"
    }
    return (
        f"A hyper-realistic portrait of a {age_descriptions[age_category]} "
        f"with {features['face_shape']} face shape, "
        f"{features['eye_color']} eyes, {features['hair_color']} hair, "
        "photorealistic, high detail, natural lighting, neutral background",
        "blurry, low quality, cartoonish, distorted, unrealistic"
    )

def generate_image_sync(features, age_category):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to(device)
    try:
        prompt, negative_prompt = create_prompt(features, age_category)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]
        return image
    finally:
        del pipe
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

@router.post("/generate-child-image")
async def generate_child_image(
    parent1_image: UploadFile = File(...),
    parent2_image: UploadFile = File(...),
    age_category: str = Form(...)
):
    try:
        img1_data = await parent1_image.read()
        img1 = cv2.imdecode(np.frombuffer(img1_data, np.uint8), cv2.IMREAD_COLOR)
        img1 = resize_image(img1)
        features1 = extract_features(img1)
        img2_data = await parent2_image.read()
        img2 = cv2.imdecode(np.frombuffer(img2_data, np.uint8), cv2.IMREAD_COLOR)
        img2 = resize_image(img2)
        features2 = extract_features(img2)
        morphed_features = morph_features(features1, features2)
        child_img = await asyncio.get_event_loop().run_in_executor(
            executor, generate_image_sync, morphed_features, age_category
        )
        img_byte_arr = io.BytesIO()
        child_img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        return Response(content=img_bytes, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")