# flask import
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
import os
import shutil
import json
import numpy as np
import torch
import io
import time
import random
import json
import cv2
from PIL import Image
import base64
import torch
from scipy.spatial import KDTree
import re
from ultralytics.models.fastsam import FastSAMPredictor
import shutil
import clip

# diffusion import
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from lib_omost.pipeline import StableDiffusionXLOmostPipeline
import lib_omost.memory_management as memory_management

# import mllm
import dashscope
import base64
from openai import OpenAI
dashscope.api_key = ''

# import word2vec 
import gensim
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(r"..\model\GoogleNews-vectors-negative300.bin", binary=True)

################################# load diffusion model ###############################################

# realistic lightning model
# hg_sdxl_name = r'SG161222/RealVisXL_V4.0_Lightning'
hg_sdxl_name = r'recoilme/ColorfulXL-Lightning'
local_sdxl_name = r'..\model\colorfulxlLightning_v16.safetensors'

# original sdxl model
# sdxl_name = 'stabilityai/stable-diffusion-xl-base-1.0'

# if only load from huggingface model, use below directly
tokenizer = CLIPTokenizer.from_pretrained(
    hg_sdxl_name, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(
    hg_sdxl_name, subfolder="tokenizer_2")
text_encoder = CLIPTextModel.from_pretrained(
    hg_sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
text_encoder_2 = CLIPTextModel.from_pretrained(
    hg_sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKL.from_pretrained(
    hg_sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16")  # bfloat16 vae
unet = UNet2DConditionModel.from_pretrained(
    hg_sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")

# if load from local model, use below. However we found local model always causing errors would not happen in huggingface model
# pipe = StableDiffusionXLPipeline.from_single_file(local_sdxl_name, torch_dtype=torch.float16)

# text_encoder = pipe.text_encoder
# text_encoder_2 = pipe.text_encoder_2
# tokenizer = pipe.tokenizer
# tokenizer_2 = pipe.tokenizer_2
# vae = pipe.vae
# unet = pipe.unet
    
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

pipeline = StableDiffusionXLOmostPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
)

memory_management.unload_all_models([text_encoder, text_encoder_2,vae, unet])
###################################################################################################

#################################### load sam #######################################
device = "cuda" if torch.cuda.is_available() else "cpu"
overrides = dict(conf=0.5, task="segment", mode="predict", model=r"..\model\mobile_sam_test\FastSAM-x.pt", save=False, imgsz=512,device = device)
predictor = FastSAMPredictor(overrides=overrides)
######################################################################################

####################################### load clip ###################################
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
#####################################################################################

@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results

@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

@torch.inference_mode()
def diffusion_fn(custom_prompts, custom_masks, num_samples, seed, image_width, image_height,
                steps, cfg, negative_prompt,custom_prompts_neg,custom_masks_neg ,highres_scale,highres_steps,highres_denoise,
    ):

    use_initial_latent = False

    image_width, image_height = int(image_width // 64) * 64, int(image_height // 64) * 64

    rng = torch.Generator(device=memory_management.gpu).manual_seed(seed)

    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])

    bag_of_conditions = [{'prefixes': p, 'suffixes': s, 'mask': m} for p, s, m in zip(custom_prompts, custom_prompts, custom_masks)]
    canvas_outputs = {'bag_of_conditions': bag_of_conditions, 'initial_latent': torch.zeros(size=(num_samples, 4, image_height // 8, image_width // 8), dtype=torch.float32)}
    
    positive_cond, positive_pooler, negative_cond, negative_pooler = pipeline.all_conds_from_canvas(canvas_outputs, negative_prompt,custom_prompts_neg,custom_masks_neg)

    if use_initial_latent:
        memory_management.load_models_to_gpu([vae])
        initial_latent = torch.from_numpy(canvas_outputs['initial_latent'])[None].movedim(-1, 1) / 127.5 - 1.0
        initial_latent_blur = 40
        initial_latent = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(initial_latent, (initial_latent_blur,) * 4, mode='reflect'),
            kernel_size=(initial_latent_blur * 2 + 1,) * 2, stride=(1, 1))
        initial_latent = torch.nn.functional.interpolate(initial_latent, (image_height, image_width))
        initial_latent = initial_latent.to(dtype=vae.dtype, device=vae.device)
        initial_latent = vae.encode(initial_latent).latent_dist.mode() * vae.config.scaling_factor
    else:
        initial_latent = torch.zeros(size=(num_samples, 4, image_height // 8, image_width // 8), dtype=torch.float32)

    memory_management.load_models_to_gpu([unet])

    initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)

    latents = pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=int(steps),
        batch_size=num_samples,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=float(cfg),
    ).images
    
    memory_management.unload_all_models([text_encoder, text_encoder_2, unet])
    memory_management.load_models_to_gpu([vae])
    
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    pixels = vae.decode(latents).sample
    memory_management.unload_all_models([vae,text_encoder, text_encoder_2, unet])

    B, C, H, W = pixels.shape
    pixels = pytorch2numpy(pixels)
    # # Here can do the high res up scale, but currently we producing 1024*1024 that does not nessesarily need it
    # eps = 0.05
    # if highres_scale > 1.0 + eps:
    #     pixels = [
    #         resize_without_crop(
    #             image=p,
    #             target_width=int(round(W * highres_scale / 64.0) * 64),
    #             target_height=int(round(H * highres_scale / 64.0) * 64)
    #         ) for p in pixels
    #     ]

    #     pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    #     latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor

    #     latents = latents.to(device=unet.device, dtype=unet.dtype)

    #     latents = pipeline(
    #         initial_latent=latents,
    #         strength=highres_denoise,
    #         num_inference_steps=highres_steps,
    #         batch_size=num_samples,
    #         prompt_embeds=positive_cond,
    #         negative_prompt_embeds=negative_cond,
    #         pooled_prompt_embeds=positive_pooler,
    #         negative_pooled_prompt_embeds=negative_pooler,
    #         generator=rng,
    #         guidance_scale=float(cfg),
    #     ).images
        
    #     latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    #     pixels = vae.decode(latents).sample
    #     pixels = pytorch2numpy(pixels)
    return pixels[0], pixels

def process_drawing(mask_input, seed):
    '''generate image without anchoring'''
    # handling and parsing input conditions from prompt json file and user input mask based on predefined color
    seed = int(seed)
    with open(r'mask_prompt.json', 'r') as f:
            data = json.load(f)

    print('generating!')

    layer_array = np.array(mask_input)

    predefined_colors = {
        (255, 0, 0): [data["red"]["type"], data["red"]["attribute"], data["red"]["state"], data["red"]["direction"],],
        (0, 255, 0): [data["green"]["type"], data["green"]["attribute"], data["green"]["state"], data["green"]["direction"],],
        (0, 0, 255): [data["blue"]["type"], data["blue"]["attribute"], data["blue"]["state"], data["blue"]["direction"],],
        (255, 255, 0): [data["yellow"]["type"], data["yellow"]["attribute"], data["yellow"]["state"], data["yellow"]["direction"],],
    }

    color_map = {
        'red': ((255, 0, 0), [data["red"]["type"], data["red"]["attribute"], data["red"]["state"], data["red"]["direction"]]),
        'green': ((0, 255, 0), [data["green"]["type"], data["green"]["attribute"], data["green"]["state"], data["green"]["direction"]]),
        'blue': ((0, 0, 255), [data["blue"]["type"], data["blue"]["attribute"], data["blue"]["state"], data["blue"]["direction"]]),
        'yellow': ((255, 255, 0), [data["yellow"]["type"], data["yellow"]["attribute"], data["yellow"]["state"], data["yellow"]["direction"]]),
        'background': ((255, 255, 255), [data["background"]["type"], data["background"]["attribute"], data["background"]["state"], data["background"]["direction"]]),
    }

    pixels = layer_array[layer_array[:, :, 3] > 0][:, :3]

    tree = KDTree(list(predefined_colors.keys()))

    _, indices = tree.query(pixels)
    closest_colors = [list(predefined_colors.keys())[index] for index in indices]

    masks = {color: np.zeros((1024, 1024)) for color in predefined_colors.keys()}

    pixel_idx = 0
    for y in range(layer_array.shape[0]):
        for x in range(layer_array.shape[1]):
            if layer_array[y, x, 3] > 0:  
                color = tuple(closest_colors[pixel_idx])
                masks[color][y, x] = 1
                pixel_idx += 1

    background_mask = np.ones((1024, 1024))
    for mask in masks.values():
        background_mask -= mask

    predefined_colors_neg = {
        (255, 0, 0): [data["red"]["negative"]],
        (0, 255, 0): [data["green"]["negative"]],
        (0, 0, 255): [data["blue"]["negative"]],
        (255, 255, 0): [data["yellow"]["negative"]],
    }
    bg_prompt_neg = [data["background"]["negative"]]
    if bg_prompt_neg == ['']:
        bg_prompt_neg = ['worst quality,low resolution']
    custom_prompts_neg = [predefined_colors_neg[color] for color in predefined_colors_neg.keys()]+ [bg_prompt_neg]
    custom_masks_neg = [masks[color] for color in predefined_colors_neg.keys()]+ [background_mask]

    bg_prompt = [data["background"]["type"], data["background"]["attribute"]]

    if bg_prompt == ['','']:
        bg_prompt = ['scene']

    custom_prompts = [predefined_colors[color] for color in predefined_colors.keys()] + [bg_prompt]
    custom_masks = [masks[color] for color in predefined_colors.keys()] + [background_mask]

    for color_key, relations in data["relationships"].items():
        for target_color, relationship in relations.items():
            if relationship:
                if color_key=='background':
                    color2 = color_map[target_color][0]
                    joint_mask = np.maximum(background_mask, masks[color2])  
                    custom_masks_neg.append(background_mask)
                    custom_prompts_neg.append(target_color)
                    custom_masks_neg.append(masks[color2])
                    custom_prompts_neg.append(color_key)
                elif target_color == 'background':
                    color1 = color_map[color_key][0]
                    joint_mask = np.maximum(masks[color1], background_mask)  
                    custom_masks_neg.append(masks[color1])
                    custom_prompts_neg.append(target_color)
                    custom_masks_neg.append(background_mask)
                    custom_prompts_neg.append(color_key)
                else:
                    color1 = color_map[color_key][0]
                    color2 = color_map[target_color][0]
                    joint_mask = np.maximum(masks[color1], masks[color2])  
                    custom_masks_neg.append(masks[color1])
                    custom_prompts_neg.append(target_color)
                    custom_masks_neg.append(masks[color2])
                    custom_prompts_neg.append(color_key)
                
                joint_prompt = [relationship]
                custom_masks.append(joint_mask)
                custom_prompts.append(joint_prompt)

    overall_mask = np.ones((1024, 1024))
    overall_prompt = [data["style"], data["camera"], data["quality"], data["lighting"], data["environment"]]
    custom_prompts.append(overall_prompt)
    custom_masks.append(overall_mask)


    image, images = diffusion_fn(
    custom_prompts=custom_prompts,
    custom_masks=custom_masks,
    num_samples=1,
    seed=random.randint(1, 999999),
    image_width=1024,
    image_height=1024,
    steps=8,
    cfg = 1.5,
    negative_prompt = "NSFW, nude, naked, ugly face, mutated hands,mutated legs, lowres, blurry face, bad anatomy, bad proportions, deformed, deformed anatomy, deformed fingers, censored, deformed, black and white, disfigured, low contrast",
    custom_prompts_neg = custom_prompts_neg,
    custom_masks_neg = custom_masks_neg,
    highres_steps=8,
    highres_denoise=0.6,
    highres_scale=1,
    )

    return image

# ######################## webui ######################################
webui_server_url = 'http://127.0.0.1:7860'
from datetime import datetime
import urllib.request

def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))

def call_api(api_endpoint, **payload):    #     json.dump(payload, f, indent=4)
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))

def call_txt2img_api(**payload):
    folder_path = '../frontend/src/assets/generated_result'
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file_path.endswith(('.png', '.png', '.jpeg', '.bmp', '.gif')):
            os.remove(file_path)

    response = call_api('sdapi/v1/txt2img', **payload)


    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(r'..\frontend\src\assets\anchor_images\overall', rf'txt2img-{timestamp()}-{index}.png')    
        save_path2 = os.path.join(r'..\frontend\src\assets\generated_result', rf'txt2img-{timestamp()}-{index}.png')        
        decode_and_save_base64(image, save_path)
        decode_and_save_base64(image, save_path2)

        
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        files.sort()

        if len(files) == 2:
            first_image = os.path.join(folder_path, files[0])
            second_image = os.path.join(folder_path, files[1])
            
            new_name = os.path.join(folder_path, "result.png")
            os.rename(first_image, new_name)
            print(f"Renamed {files[0]} to {new_name}")
            
            os.remove(second_image)
            print(f"Deleted {files[1]}")
        else:
            print("The folder does not contain exactly two images.")

def process_drawing_with_anchor(seed, anchor_img, anchor_seg):
    '''generate image with anchor using regional prompter and controlnet'''

    # parse mask / prompt
    seed = int(seed)
    with open(r'mask_prompt.json', 'r') as f:
            data = json.load(f)

    def is_valid_prompt(prompt):
        return any(part.strip() for part in prompt.split(','))
    
    color_map = {
        'red': ((255, 0, 0), ','.join([data["red"]["type"], data["red"]["attribute"], data["red"]["state"], data["red"]["direction"]])),
        'green': ((0, 255, 0), ','.join([data["green"]["type"], data["green"]["attribute"], data["green"]["state"], data["green"]["direction"]])),
        'blue': ((0, 0, 255), ','.join([data["blue"]["type"], data["blue"]["attribute"], data["blue"]["state"], data["blue"]["direction"]])),
        'yellow': ((255, 255, 0), ','.join([data["yellow"]["type"], data["yellow"]["attribute"], data["yellow"]["state"], data["yellow"]["direction"]])),
    }

    overall_prompt = ','.join([data["style"], data["camera"], data["quality"], data["lighting"], data["environment"]])
    bg_prompt = ','.join([data["background"]["type"], data["background"]["attribute"]])
    overall_prompt = overall_prompt + ',' + bg_prompt 

    prompt_red = color_map['red'][1]
    prompt_green = color_map['green'][1]
    prompt_blue = color_map['blue'][1]
    prompt_yellow = color_map['yellow'][1]

    for color_key, relations in data["relationships"].items():
        for target_color, relationship in relations.items():
            if relationship and color_key != 'background' and target_color != 'background':
                if color_key == 'red' and target_color == 'green':
                    prompt_red += f",{relationship}"
                elif color_key == 'green' and target_color == 'red':
                    prompt_green += f",{relationship}"
                elif color_key == 'red' and target_color == 'blue':
                    prompt_red += f",{relationship}"
                elif color_key == 'blue' and target_color == 'red':
                    prompt_blue += f",{relationship}"
                elif color_key == 'red' and target_color == 'yellow':
                    prompt_red += f",{relationship}"
                elif color_key == 'yellow' and target_color == 'red':
                    prompt_yellow += f",{relationship}"
                elif color_key == 'green' and target_color == 'blue':
                    prompt_green += f",{relationship}"
                elif color_key == 'blue' and target_color == 'green':
                    prompt_blue += f",{relationship}"
                elif color_key == 'green' and target_color == 'yellow':
                    prompt_green += f",{relationship}"
                elif color_key == 'yellow' and target_color == 'green':
                    prompt_yellow += f",{relationship}"
                elif color_key == 'blue' and target_color == 'yellow':
                    prompt_blue += f",{relationship}"
                elif color_key == 'yellow' and target_color == 'blue':
                    prompt_yellow += f",{relationship}"


    # prepare for regional prompter 
    prompt_parts = [overall_prompt + " ADDCOMM"]
    color_prompts = []
    if is_valid_prompt(prompt_red):
        color_prompts.append(prompt_red)
    if is_valid_prompt(prompt_green):
        color_prompts.append(prompt_green)
    if is_valid_prompt(prompt_blue):
        color_prompts.append(prompt_blue)
    if is_valid_prompt(prompt_yellow):
        color_prompts.append(prompt_yellow)

    if color_prompts:
        prompt = ' ADDCOMM '.join([overall_prompt, ' BREAK '.join(color_prompts)])
    else:
        prompt = overall_prompt + " ADDCOMM"
    neg_prompt = 'worst quality, low resolution'

    # this payload use api to call regional propmter and controlnet
    payload = {
        "prompt": f'''{prompt}''',  # extra networks also in prompts
        "negative_prompt": f"{neg_prompt}",
        "seed": random.randint(1, 999999),
        "steps": 9,
        "width": 1024,
        "height": 1024,
        "cfg_scale": 1,
        "sampler_name": "Euler a",
        "n_iter": 1,
        "batch_size": 1,
        # example args for Refiner and ControlNet
        "alwayson_scripts": {
            "ControlNet": {
                "args": [
                    {
                        "batch_images": "",
                        "control_mode": "My prompt is more important",
                        "enabled": True,
                        "guidance_end": 0.4,
                        "guidance_start": 0,
                        "image": {
                            "image": encode_file_to_base64(anchor_img),
                            "mask": None  # base64, None when not need
                        },
                        "input_mode": "simple",
                        "is_ui": True,
                        "loopback": False,
                        "low_vram": False,
                        "model": "diffusers_xl_canny_full [2b69fca4]",
                        "module": "canny",
                        "output_dir": "",
                        "pixel_perfect": False,
                        "processor_res": 512,
                        "resize_mode": "Crop and Resize",
                        "threshold_a": 100,
                        "threshold_b": 200,
                        "weight": 0.8
                    }
                ]
            },
            "Regional Prompter": {
			"args":	[
                True, # active
                False, # debug
                "Mask",  # mode (matrix,mask,prompt)
                "Vertical", # mode (matrix)
                "Mask", # Mode (Mask)
                "Prompt", # Mode (Prompt)
                "1,1,1", # Ratios
                "", # Base Ratios
                False, # base
                True, # common
                False, # neg common
                "Attention", 
                False, # Not Change AND
                "0", # LoRA Textencoder
                "0", # LoRA U-Net
                "0", # Threshold
                anchor_seg, # Mask
                ]
            },
        },
        "denoising_strength": 0.5,
        "override_settings": {
            'sd_model_checkpoint': r"..\model\colorfulxlLightning_v16.safetensors",  # this can use to switch sd model
        },
    }
    result = call_txt2img_api(**payload)
    return 

def process_drawing_single(mask_input, seed, object_color):
    '''generate single object image'''
    seed = int(seed)
    with open(r'mask_prompt.json', 'r') as f:
            data = json.load(f)

    print('generating!')

    layer_array = np.array(mask_input)

    predefined_colors = {
        (255, 0, 0): [data["red"]["type"], data["red"]["attribute"], data["red"]["state"],],
        (0, 255, 0): [data["green"]["type"], data["green"]["attribute"], data["green"]["state"],],
        (0, 0, 255): [data["blue"]["type"], data["blue"]["attribute"], data["blue"]["state"],],
        (255, 255, 0): [data["yellow"]["type"], data["yellow"]["attribute"], data["yellow"]["state"],],
    }

    rgb_to_color_name = {
        (255, 0, 0): "red",
        (0, 255, 0): "green",
        (0, 0, 255): "blue",
        (255, 255, 0): "yellow"
    }

    type_map = {
        (255, 0, 0): data["red"]["type"],
        (0, 255, 0): data["green"]["type"],
        (0, 0, 255): data["blue"]["type"],
        (255, 255, 0): data["yellow"]["type"],
    }

    # classify object type based on coco stuff object
    with open(r'coco_stuff_object.json', 'r') as f:
        coco_data = json.load(f)

    def is_object(type_value):
        max_thing_similarity = float('-inf')
        max_stuff_similarity = float('-inf')

        for thing in coco_data["object"]:
            try:
                similarity = word2vec_model.similarity(type_value, thing)
                if similarity > max_thing_similarity:
                    max_thing_similarity = similarity
            except KeyError:
                continue 


        for stuff in coco_data["stuff"]:
            try:
                similarity = word2vec_model.similarity(type_value, stuff)
                if similarity > max_stuff_similarity:
                    max_stuff_similarity = similarity
            except KeyError:
                continue  

        return max_thing_similarity < max_stuff_similarity
    

    pixels = layer_array[layer_array[:, :, 3] > 0][:, :3]

    tree = KDTree(list(predefined_colors.keys()))

    _, indices = tree.query(pixels)
    closest_colors = [list(predefined_colors.keys())[index] for index in indices]

    masks = {color: np.zeros((512, 512)) for color in predefined_colors.keys()}

    pixel_idx = 0
    for y in range(layer_array.shape[0]):
        for x in range(layer_array.shape[1]):
            if layer_array[y, x, 3] > 0:  
                color = tuple(closest_colors[pixel_idx])
                masks[color][y, x] = 1
                pixel_idx += 1

    for color in masks.keys():
        mask = masks[color]
        mask = cv2.erode(mask, None, iterations=2)  
        mask = cv2.dilate(mask, None, iterations=2)  
        masks[color] = mask

    background_mask = np.ones((512, 512))
    for mask in masks.values():
        background_mask -= mask


    bg_prompt = [data["background"]["type"], data["background"]["attribute"]]
    if bg_prompt == ['','']:
        bg_prompt = ['background']

    custom_prompts = []
    custom_masks = []
    
    for color, prompt in predefined_colors.items():
        color_name = rgb_to_color_name.get(color)  
        if is_object(type_map[color]) or color_name == object_color:  
            custom_prompts.append(prompt)
            custom_masks.append(masks[color])
        else:
            print(f"Skipping generation for {color} as it is classified as object.")

    custom_prompts.append(bg_prompt)
    custom_masks.append(background_mask)

    front_end_output_mask_folder = r'..\frontend\src\assets\mask'
    for color, mask in masks.items():

        mask_img = Image.fromarray((mask * 255).astype(np.uint8))  
        mask_img = mask_img.convert("L")  


        color_name = rgb_to_color_name.get(color, f"{color[0]}_{color[1]}_{color[2]}")


        for file in os.listdir(front_end_output_mask_folder):
            if file.startswith(color_name) and file.endswith('.png'):
                os.remove(os.path.join(front_end_output_mask_folder, file))
                print(f"Deleted old mask file: {file}")

        mask_indices = np.where(mask > 0)
        if mask_indices[0].size > 0 and mask_indices[1].size > 0:
            y_min, y_max = np.min(mask_indices[0]), np.max(mask_indices[0])
            x_min, x_max = np.min(mask_indices[1]), np.max(mask_indices[1])
            bbox = (x_min, y_min, x_max, y_max)
            bbox_str = f"{x_min}_{y_min}_{x_max}_{y_max}"
            print(f"Bounding box for color {color_name}: {bbox}")
        else:
            bbox_str = "no_bbox"
            print(f"No non-transparent pixels found for color {color_name}, skipping bounding box calculation.")

        file_name = f"{color_name}_{bbox_str}.png"
        mask_path = os.path.join(front_end_output_mask_folder, file_name)
        mask_img.save(mask_path)
        print(f"Saved mask for color {color_name} with bounding box {bbox_str} to {mask_path}")

    # use 512*512 for fast generation and save memory
    overall_mask = np.ones((512, 512))
    overall_prompt = [data["style"], data["camera"], data["quality"], data["lighting"], data["environment"]]
    custom_prompts.append(overall_prompt)
    custom_masks.append(overall_mask)

    predefined_colors_neg = {
        (255, 0, 0): [data["red"]["negative"]],
        (0, 255, 0): [data["green"]["negative"]],
        (0, 0, 255): [data["blue"]["negative"]],
        (255, 255, 0): [data["yellow"]["negative"]],
    }
    bg_prompt_neg = [data["background"]["negative"]]
    if bg_prompt_neg == ['']:
        bg_prompt_neg = ['worst quality,low resolution']
    custom_prompts_neg = [predefined_colors_neg[color] for color in predefined_colors_neg.keys()]+ [bg_prompt_neg]
    custom_masks_neg = [masks[color] for color in predefined_colors_neg.keys()]+ [background_mask]
    
    image, images = diffusion_fn(
    custom_prompts=custom_prompts,
    custom_masks=custom_masks,
    num_samples=4,
    seed=random.randint(1, 999999),
    image_width=512,
    image_height=512,
    steps=8,
    cfg = 1,
    negative_prompt = "NSFW, nude, naked, ugly face, mutated hands,mutated legs, lowres, blurry face, bad anatomy, bad proportions, deformed, deformed anatomy, deformed fingers, censored,deformed, black and white, disfigured, low contrast",
    custom_prompts_neg = custom_prompts_neg,
    custom_masks_neg = custom_masks_neg,
    highres_steps=8,
    highres_denoise=0.6,
    highres_scale=1,
    )

    return image, images

def infer_semantic(drawing):
    '''inference prompts for each object'''
    # templates for inference
    example_space = '''
    {
        "camera": "",
        "background": {
            "type": "",
            "attribute": ""
        },
        "red": {
            "type": "",
            "attribute": "",
            "state": "",
            "direction": ""
        },
        "green": {
            "type": "",
            "attribute": "",
            "state": "",
            "direction": ""
        },
        "blue": {
            "type": "",
            "attribute": "",
            "state": "",
            "direction": ""
        },
        "yellow": {
            "type": "",
            "attribute": "",
            "state": "",
            "direction": ""
        },
        "relationships": {
            "red": {
                "green": "",
                "blue": "",
                "yellow": "",
                "background": ""
            },
            "green": {
                "red": "",
                "blue": "",
                "yellow": "",
                "background": ""
            },
            "blue": {
                "red": "",
                "green": "",
                "yellow": "",
                "background": ""
            },
            "yellow": {
                "red": "",
                "green": "",
                "blue": "",
                "background": ""
            }
        }
    }
    '''
    few_shot = '''
        "camera": "close-up",
        "background": {
            "type": "living room",
            "attribute": "cozy corner, furnished, warm, well-lit"
        },
        "red": {
            "type": "girl",
            "attribute": "wearing casual indoor clothing, gentle expression, brown hair, blue eyes, youthful, full body",
            "state": "kneeling, smiling",
            "direction": "girl facing the cat"
        },
        "green": {
            "type": "cat",
            "attribute": "small, furry, black fur, curled tail, attentive",
            "state": "sitting comfortably",
            "direction": "cat facing and slightly turned towards the girl"
        },
        "relationships": {
            "red": {
                "green": "girl petting with cat",
                "blue": "",
                "yellow": "",
                "background": ""
            },
            "green": {
                "red": "cat being petted by girl",
                "blue": "",
                "yellow": "",
                "background": ""
            }
        }

        "camera": "medium shot",
        "background": {
            "type": "distant mountainous landscape",
            "attribute": "clear sky, sunset glow, natural setting, tranquil atmosphere",
            "state": "",
            "direction": "",
            "negative": ""
        },
        "red": {
            "type": "girl",
            "attribute": "long black hair, white dress, calm, bare foot, slim shape, full body",
            "state": "one arm extent, standing",
            "direction": "girl facing slightly towards the boy",
            "negative": ""
        },
        "green": {
            "type": "boy",
            "attribute": "short black hair, wearing casual t-shirt, excited expression, slightly tanned skin, bare foot. full body",
            "state": "standing, one arm extent",
            "direction": "boy facing slightly towards the ocean",
            "negative": ""
        },
        "blue": {
            "type": "ocean",
            "attribute": "calm, wide, reflective surface, gently rippling, vast",
            "state": "still",
            "direction": "ocean extending across the lower part of the scene",
            "negative": ""
        },
        "yellow": {
            "type": "mountain",
            "attribute": "distant, large, majestic, rugged, partially obscured by mist",
            "state": "stationary",
            "direction": "background, far right",
            "negative": ""
        },
        "relationships": {
            "red": {
                "green": "girl holding hands with boy, standing next to each other",
                "blue": "girl standing near the ocean",
                "yellow": "girl facing towards the ocean and distant mountain",
                "background": "girl standing on a beach with mountain in the distance"
            },
            "green": {
                "red": "boy holding hands with girl, standing next to each other",
                "blue": "boy standing near the ocean",
                "yellow": "boy facing towards the ocean and distant mountain",
                "background": "boy standing on a beach with mountain in the distance"
            },
            "blue": {
                "red": "ocean close to the girl, extending along the bottom of the scene",
                "green": "ocean close to the boy, extending along the bottom of the scene",
                "yellow": "ocean leading towards the distant mountain",
                "background": "ocean at the beach with mountain in the background"
            },
            "yellow": {
                "red": "mountain far behind the girl, part of the background",
                "green": "mountain far behind the boy, part of the background",
                "blue": "mountain visible beyond the ocean, part of the background",
                "background": "mountain as a distant background feature"
            }
        }
    '''

    with open(r'mask_prompt.json', 'r') as f:
        data = json.load(f)   
    with open(r'dataset/attribute.json', 'r') as f:
        attribute_data =  json.load(f) 
    with open(r'dataset/relationship.json', 'r') as f:
        relationship_data =  json.load(f) 

    def save_temp_image(image):
        # image = Image.fromarray(image_array.astype('uint8'), 'RGB')
        image_path = "temp_image.png"
        image.save(image_path)
        return image_path
    # Open the image file and encode it as a base64 string
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8") 
    
    def update_dict(original, new):
        for key, value in new.items():
            if key == "background" and isinstance(value, dict):
                orig_bg = original.get("background", {})
                new_bg = value

                if not orig_bg.get("type", "").strip():
                    orig_bg["type"] = new_bg.get("type", "")

                if not orig_bg.get("attribute", "").strip():
                    orig_bg["attribute"] = new_bg.get("attribute", "")
                original["background"] = orig_bg
                continue
            if isinstance(value, dict) and key in original:
                update_dict(original[key], value)
            else:
                original[key] = value 

    # Clean the assistant reply to ensure it's valid JSON
    def clean_json_string(json_string):
        # Remove any leading/trailing whitespace and unnecessary markers
        json_string = json_string.strip()
        
        json_string = re.findall(
            r'```(.*?)```', assistant_reply, re.DOTALL)

        json_string = json_string[0].strip()
        if json_string.startswith('json'):
            json_string = json_string.lstrip('json')
        
        return json_string

    output_format = example_space
    output_dict = json.loads(output_format)
    colors = ["red", "green", "blue", "yellow"]
    input_color_mask = ''
    attributes = ''
    relationships_list = []
    relationships_str = ""  

    for color in colors:
        color_data = data.get(color, {})
        color_type = color_data.get("type", "")
        
        if color_type:
            output_dict[color]["type"] = color_type
            input_color_mask = input_color_mask + ' the ' + color + ' mask is ' + color_type + ';'

            for attr in attribute_data:
                if attr.get("object_name") == color_type:
                    positive_attrs = attr.get("positive_attributes", [])
                    if len(positive_attrs) > 5:
                        attrs = random.sample(positive_attrs, 5)
                    else:
                        attrs = positive_attrs
                    break  
                else:
                    attrs = []
            output_dict[color]["attributes"] = attrs
            attributes += f' the {color_type} has attributes {", ".join(attrs)}' + ';'
        else:
            del output_dict[color]    

    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            type1 = data.get(colors[i], {}).get("type", "")
            type2 = data.get(colors[j], {}).get("type", "")
            if type1 and type2:
                current_predicates = []
                # 针对 type2 到 type1 的关系
                for rel in relationship_data:
                    if (rel.get("object_name") == type1 and rel.get("subject_name") == type2):
                        predicates_list = rel.get("predicates", [])
                        if len(predicates_list) > 5:
                            current_predicates.extend(random.sample(predicates_list, 5))
                        else:
                            current_predicates.extend(predicates_list)
                        break  

                if current_predicates:
                    relationships_str += f'{type2} to {type1}: {", ".join(current_predicates)}\n' + ';'
                else:
                    relationships_str += f'{type2} to {type1}: None\n'

                current_predicates = []

                for rel in relationship_data:
                    if (rel.get("object_name") == type2 and rel.get("subject_name") == type1):
                        predicates_list = rel.get("predicates", [])
                        if len(predicates_list) > 5:
                            current_predicates.extend(random.sample(predicates_list, 5))
                        else:
                            current_predicates.extend(predicates_list)
                        break  

                if current_predicates:
                    relationships_str += f'{type1} to {type2}: {", ".join(current_predicates)}\n' + ';'
                else:
                    relationships_str += f'{type1} to {type2}: None\n'

    updated_output_str = json.dumps(output_dict, indent=4)

    question = f''' 
        Here is a sketch of an image. 
        {input_color_mask}, the rest of white space is background. 
        It might be a very rough sketch,the mask are approximate range of the object shape, but I need you to infer some details of the image based on the the given sketch.
        The details include the possible background that mostly to be given {input_color_mask} present in a scene, the attribute of each object(atleast 5 attributes like wearing, texture, color, full body/half body/head shots etc., do not include anything related to other object and do not include any action), the state(include action,posture etc., do not include anything related to other object) of each object, the direction of each object, the relationship between objects, the camera of the whole image.
        You should first analyze the mask carefully, especially consider the size, location and relative position of each object mask, make sure the specific action is analyzed based on the mask (raising hands, extending arms, jumping in the air or simply standing etc.), infer each aspect to be filled with reasoning process, then you give the final result output.
        Print your reasoning process before your final output. 
        The final output format should be: {updated_output_str}, you should refer to the example: {few_shot}. You are going to complete the "" in each item, you need to complete them in multiple short phrases based on your above reasoning.
        Do not consider the color of mask, background is white does not mean background is related to white, is just a space that you need to figure out what is should be.
        The state and relationship should be as detailed as possible but also make sure it align with the mask.
        The relationship must include action and spatial relationship, and should be in format: objectA action/spatial relation objectB, and the object A,objectB should be included, e.g. (girl petting the cat), that both girl(objectA) and cat(objectB) are included in the relationship.
        You should properly refer to some examples of attributes of object {attributes} and relationships {relationships_list}.
        Do not include words like 'or', 'possibily' in your final output, once you finish your final output, check if there is anything like 'or','possibily' in your output, and if any color that I didn't given to you but you still put in your output, there should no ambiguity in your output.
        Make sure all aspects of given mask is filled.
    ''' 

    client = OpenAI(base_url='https://api.chsdw.top/v1',
        api_key='sk-VsP39aCKuvCR5f2E808aAfE202904c68BbB4DfDa68Dc3a2b')    
    
    image_path = save_temp_image(drawing)
    base64_image = encode_image(image_path)

    messages = [
        {"role": "system", "content": f"You are a highly intelligent AI familiar scene description."},
        {"role": "user", 
         "content": [
           {
               "type": "text", 
               "text": f"{question}"
            },
           {
               "type": "image_url",
               "image_url":{
                   "url": f"data:image/png;base64,{base64_image}"
                }
           }
         ]
        }
    ]
    try:
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-16k",
            model="gpt-4o-2024-11-20",
            messages=messages
        )
    except:
        return
    try:
        assistant_reply = response.choices[0].message.content
        print(assistant_reply)
    except Exception as e:
        print(f"Error processing : {e}")

    cleaned_reply = clean_json_string(assistant_reply)
    # Parse the cleaned JSON string
    try:
        new_data = json.loads(cleaned_reply)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        new_data = {}

    update_dict(data, new_data)
    # Step 4: Save the updated data back to the JSON file
    with open(r'mask_prompt.json', 'w') as f:
        json.dump(data, f, indent=4)
    print("Data has been updated successfully.")

    return

# fetch object and correlated semantics
def initialize_mask_prompt():
    all_semantics = {
        "style": "",
        "camera": "",
        "quality": "",
        "lighting": "",
        "environment": "",
        "background": {"type": "", "attribute": "", "state": "", "direction": "","negative": ""},
        "red": {"type": "", "attribute": "", "state": "", "direction": "","negative": ""},
        "green": {"type": "", "attribute": "", "state": "", "direction": "","negative": ""},
        "blue": {"type": "", "attribute": "", "state": "", "direction": "","negative": ""},
        "yellow": {"type": "", "attribute": "", "state": "", "direction": "","negative": ""},
        "relationships": {
            "red": {"green": "", "blue": "", "yellow": "", "background": ""},
            "green": {"red": "", "blue": "", "yellow": "", "background": ""},
            "blue": {"red": "", "green": "", "yellow": "", "background": ""},
            "yellow": {"red": "", "green": "", "blue": "", "background": ""}
        }
    }

    with open(r'mask_prompt.json', 'w') as f:
        json.dump(all_semantics, f, indent=4)

    return 'all cleaned!'

def combine_images_overall():
    base_path = r"../frontend/src/assets/anchor_images/"
    mask_alternate_path = r"../frontend/src/assets/mask/"
    overall_save_path = os.path.join(base_path, "overall")
    os.makedirs(overall_save_path, exist_ok=True)

    cropped_images = []
    mask_images = []

    # Iterate over each color folder
    for color in ['red', 'green', 'yellow', 'blue']:
        color_folder = os.path.join(base_path, color)
        if not os.path.isdir(color_folder):
            continue

        cropped_output_path = os.path.join(color_folder, 'anchor_img_cropped_output.png')
        mask_output_path = os.path.join(color_folder, 'anchor_img_mask_output.png')

        # Add the images to the list if they exist
        if os.path.exists(cropped_output_path):
            cropped_images.append(cropped_output_path)
        if os.path.exists(mask_output_path):
            mask_images.append(mask_output_path)
        else:
            # Look for an alternate mask file in the mask directory
            alternate_mask_files = [f for f in os.listdir(mask_alternate_path) if f.startswith(color) and f.endswith('.png')]
            if alternate_mask_files:
                alternate_mask_path = os.path.join(mask_alternate_path, alternate_mask_files[0])
                mask_images.append(alternate_mask_path)

    # Combine the cropped output images without color assignment
    if cropped_images:
        combined_cropped_image = combine_images_without_color(cropped_images,mask_images)
        combined_cropped_image.save(os.path.join(overall_save_path, 'combined_cropped_output.png'))
    

    # Combine the mask output images with color assignment
    if mask_images:
        combined_mask_image = assign_colors_to_masks(mask_images)
        # combined_mask_image.save(os.path.join(overall_save_path, 'combined_mask_output.png'))
        # cv2.imwrite(os.path.join(overall_save_path, 'combined_mask_output.png'), combined_mask_image)

def combine_images_without_color(image_paths, mask_paths):
    # Use with statement to open the first image
    with Image.open(image_paths[0]) as base_image:
        width, height = base_image.size

    # Create a blank image to store the combined image with a black background
    combined_image = Image.new('RGB', (width, height), (0, 0, 0))

    min_length = min(len(image_paths), len(mask_paths))
    image_paths = image_paths[:min_length]
    mask_paths = mask_paths[:min_length]

    # Reverse the order of image_paths and mask_paths for backwards stacking
    reversed_image_paths = image_paths[::-1]
    reversed_mask_paths = mask_paths[::-1]

    # Iterate over each image and its corresponding mask
    for img_path, mask_path in zip(reversed_image_paths, reversed_mask_paths):
        with Image.open(img_path) as img, Image.open(mask_path) as mask:
            img = img.convert("RGBA")  # Convert to RGBA to ensure it has an alpha channel
            mask = mask.convert("L")   # Convert mask to grayscale (L mode)

            # Debug: Print out some information about the mask and image
            # print(f"Processing {img_path} with mask {mask_path}")
            # print(f"Image size: {img.size}, Mask size: {mask.size}")


            # Extract the region of interest using the mask
            img_masked = Image.composite(img, Image.new('RGBA', img.size), mask)
            
            # Paste the masked image onto the combined image
            combined_image = Image.alpha_composite(combined_image.convert("RGBA"), img_masked)

    return combined_image.convert("RGB")  # Convert back to RGB for saving

def assign_colors_to_masks(mask_paths):
    predefined_hues = [0, 180, 90, 270]
    num_masks = len(mask_paths)
    
    if num_masks > len(predefined_hues):
        raise ValueError("Number of masks exceeds the predefined hues available.")
    
    sample_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
    target_size = (sample_mask.shape[1], sample_mask.shape[0])

    # Create a blank canvas with the same size as the masks
    combined_mask = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
    

    # Reverse the mask paths to ensure backward stacking of colors
    reversed_mask_paths = mask_paths[::-1]
    
    for idx, mask_path in enumerate(reversed_mask_paths):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert the mask to binary to ensure clean edges
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)



        # The color assignment is based on the original index order to keep the color order unchanged
        hue = predefined_hues[num_masks - 1 - idx] / 2
        hsv_color = np.uint8([[[hue, 128, 128]]])  
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

        # save individual
        colored_mask = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
        colored_mask[binary_mask == 255] = bgr_color

        # Save the individual colored mask
        if 'anchor_img_mask_output' in mask_path:
            base_name = os.path.splitext(mask_path)[0]
            output_path = f"{base_name}_with_color.png"
        else:
            # Case 2: Save in the anchor_images folder with a modified name
            # Extract the folder name (e.g., 'blue' from the mask_path)
            folder_name = os.path.basename(mask_path).split('_')[0]
            output_folder = os.path.join(
                os.path.dirname(mask_path).replace('mask', 'anchor_images'), 
                folder_name
            )
            os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists
            output_path = os.path.join(output_folder, 'anchor_img_mask_output_with_color.png')

        cv2.imwrite(output_path, colored_mask)
        # Apply the color only to the binary mask region
        mask_indices = binary_mask == 255
        combined_mask[mask_indices] = bgr_color

    return combined_mask

gallery_files=[["0_0.png", "0_1.png", "0_2.png", "0_3.png"]]
result_files=[["result.png"],['combined_mask_output.png']]

FILE_ABS_PATH = os.path.dirname(__file__)

app = Flask(__name__)
CORS(app)

@app.route('/api/test/hello/', methods=['POST'])
def hello_resp():
    params = request.json
    return "hello VUE"

def file2img(img_file):
    imgData=img_file.read()
    byte_stream = io.BytesIO(imgData)  
    im = Image.open(byte_stream) 
    return im

@app.route('/api/test/getSketch/',methods=['GET','POST'])
def getSketch():
    draw_f = request.files["sketch_draw"]

    draw_img = file2img(draw_f)
    
    draw_img = draw_img.resize((1024,1024))

    anchor_img_path = r"../frontend/src/assets/anchor_images/overall/combined_cropped_output.png"
    anchor_semantic_path = r"../frontend/src/assets/anchor_images/overall/combined_mask_output.png"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if os.path.exists(anchor_img_path):
        # if anchor exists，condcut process_drawing_with_anchor
        generated_image = process_drawing_with_anchor(random.randint(1, 999999), anchor_img_path,anchor_semantic_path)
    else:
        # else conduct process_drawing
        generated_image = process_drawing(draw_img, random.randint(1, 999999))
        generated_image = Image.fromarray(generated_image)
        save_path = os.path.join(r'..\frontend\src\assets\generated_result', 'result.png')
        generated_image.save(save_path)

    return jsonify({"error": 1001, "msg": "上传失败"})

@app.route('/api/test/getJson/',methods=['GET','POST'])
def getJson():
    if request.method == "POST":
        data = request.get_data().decode("utf-8")
        result = json.loads(data)
        with open(r"mask_prompt.json", "w") as json_file:
            json.dump(result, json_file, indent=4)
    return result

# update json instance
@app.route('/api/test/updateJson/',methods=['GET','POST'])
def updateJson():
    if request.method == "POST":
        with open(r"mask_prompt.json", "r") as json_file:
            json_file = json.load(json_file)
    # return jsonify({"error": 1001, "msg": "上传失败"})
    return jsonify(json_file)

@app.route('/api/test/getGallery/',methods=['GET','POST'])
def getGallery():
    color = request.form["color_info"]
    color = str(color).strip().strip('"').strip("'")

    image_folder_path = fr'..\frontend\src\assets\pre_gen_cache\{color}'
    all_files = os.listdir(image_folder_path)
    filtered_files = [f for f in all_files if f.endswith('.png')]
    gallery_files = [filtered_files]

    source_list=[fr"./src/assets/pre_gen_cache/{color}/" + v for v in gallery_files[0]]
    source_str="----".join(source_list)

    return jsonify({"error": 1001, "msg": "上传失败", "file_names": source_str})


@app.route('/api/test/getResult/',methods=['GET','POST'])
def getResult():
    is_preview = request.form["if_preview"]
    is_preview = str(is_preview).strip().strip('"').strip("'")
    if is_preview == 'preview':
        source_list=[fr"./src/assets/anchor_images/overall/" + v for v in result_files[1]]
    else:
        source_list=[fr"./src/assets/generated_result/" + v for v in result_files[0]]
    source_str="----".join(source_list)
    return jsonify({"error": 1001, "msg": "上传失败", "file_names": source_str})


@app.route('/api/test/getAnchorSingle/',methods=['GET','POST'])
def getAnchorSingle():
    color = request.form["color_info"]
    color = str(color).strip().strip('"').strip("'")
    anchor_img = request.form["anchor_image_info"]
    anchor_img = str(anchor_img).strip().strip('"').strip("'")
    anchor_img = os.path.basename(anchor_img)
    anchor_img = anchor_img.split('?')[0]

    # Paths for the images
    read_path = fr"../frontend/src/assets/pre_gen_cache/{color}/"
    read_mask_path = fr"../frontend/src/assets/mask/{color}/"
    
    # Construct the full paths for the images
    anchor_img_path = os.path.join(read_path, anchor_img)
    mask_img1_path = os.path.join(read_mask_path, anchor_img.replace('.png', '_cropped_output.png'))
    mask_img2_path = os.path.join(read_mask_path, anchor_img.replace('.png', '_mask_output.png'))

    # Load the images
    try:
        with Image.open(anchor_img_path) as anchor_image, \
             Image.open(mask_img1_path) as mask_image1, \
             Image.open(mask_img2_path) as mask_image2:
             
            # Save the images to the specified folder
            save_path = fr"../frontend/src/assets/anchor_images/{color}/"
            os.makedirs(save_path, exist_ok=True)  # Create the folder if it doesn't exist
            
            # Save the anchor image
            anchor_image_filename = 'anchor_img.png'
            anchor_image.save(os.path.join(save_path, anchor_image_filename))

            # Save the mask images
            mask_image1_filename = anchor_image_filename.replace('.png', '_cropped_output.png')
            mask_image2_filename = anchor_image_filename.replace('.png', '_mask_output.png')
            
            mask_image1.save(os.path.join(save_path, mask_image1_filename))
            mask_image2.save(os.path.join(save_path, mask_image2_filename))
        
        # Combine the images across all color folders
        combine_images_overall()

        return jsonify({"error": 1001, "msg": "上传失败"})

    except Exception as e:
        return jsonify({"error": 1002, "msg": f"Image loading failed: {str(e)}"})

# 做语义inference
@app.route('/api/test/getInference/',methods=['GET','POST'])
def getInference():
    draw_f = request.files["sketch_draw"]
    draw_img = file2img(draw_f)
    draw_img = draw_img.resize((1024,1024))

    infer_semantic(draw_img)
    return jsonify({"error": 1001, "msg": "上传失败"})

# 清空语义
@app.route('/api/test/getClean/',methods=['GET','POST'])
def getClean():
    def delete_all_files_in_folder(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error occurred while deleting file {file_path}: {e}")
    try:
        os.remove(r"..\frontend\src\assets\anchor_images\overall\combined_cropped_output.png")
        os.remove(r"..\frontend\src\assets\anchor_images\overall\combined_mask_output.png")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error occurred while deleting files: {e}")

    # delete blue, green, red, yellow folders
    base_folder_path = r"..\frontend\src\assets\anchor_images"
    folders_to_clean = ['blue', 'green', 'red', 'yellow']

    for folder in folders_to_clean:
        folder_path = os.path.join(base_folder_path, folder)
        delete_all_files_in_folder(folder_path)

    # 初始化mask prompt
    initialize_mask_prompt()

    # 返回响应
    return jsonify({"error": 1001, "msg": "上传失败"})

# 清空语义
@app.route('/api/test/getCleanMask/',methods=['GET','POST'])
def getCleanMask():
    print('removing anchor mask')
    try:
        os.remove(r"..\frontend\src\assets\anchor_images\overall\combined_cropped_output.png")
        os.remove(r"..\frontend\src\assets\anchor_images\overall\combined_mask_output.png")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error occurred while deleting files: {e}")

    # 返回响应
    return jsonify({"error": 1001, "msg": "上传失败"})


# 预生成单object的图
@app.route('/api/test/getGenSingle/',methods=['GET','POST'])
def getGenSingle():
    # get choosen color to gen
    color = request.form["color_info"]
    color = str(color).strip().strip('"').strip("'")

    # the mask
    draw_f = request.files["sketch_draw"]
    draw_img = file2img(draw_f)
    draw_img = draw_img.resize((512,512))

    # gen single object images
    for i in range(0,2):
        generated_image,generated_images = process_drawing_single(draw_img, random.randint(1, 999999), color)
        for j, generated_image in enumerate(generated_images):
            generated_image = Image.fromarray(generated_image)
            generated_image.save(fr'..\frontend\src\assets\pre_gen_cache\{color}\{i}_{j}.png')

    # do batch sam
    image_folder_path = fr'..\frontend\src\assets\pre_gen_cache\{color}'

    everything_results = predictor(image_folder_path)
    mask_folder_path = r'..\frontend\src\assets\mask'
    with open(r"mask_prompt.json", "r") as json_file:
        json_file = json.load(json_file)

    bboxes = []
    texts= json_file[color]["type"]

    for filename in os.listdir(mask_folder_path):
        if filename.startswith(color):
            # Remove file extension and split to extract bounding box coordinates
            base_name = os.path.splitext(filename)[0]
            parts = base_name.split('_')
            if len(parts) == 5:  # color, x1, y1, x2, y2
                x1, y1, x2, y2 = map(int, parts[1:])
                bboxes.append([x1, y1, x2, y2])

    bbox_results = predictor.prompt(everything_results, bboxes=bboxes, texts=texts, clip_model = clip_model, clip_preprocess = preprocess)

    # # Save the masks to a specified folder
    save_folder_path = fr'..\frontend\src\assets\mask\{color}'

    for i, bbox_result in enumerate(bbox_results):
        image_path = os.path.join(image_folder_path, os.listdir(image_folder_path)[i])
        image = cv2.imread(image_path)

        mask_object = bbox_result.masks  # get masks
        mask = mask_object.data[0].cpu().numpy()  

        # Apply mask to the original image
        masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

        # Save the mask
        basename = os.path.basename(os.listdir(image_folder_path)[i])
        basename = os.path.splitext(basename)[0]

        save_path = os.path.join(save_folder_path, f'{basename}_mask_output.png')
        cv2.imwrite(save_path, mask * 255)  # Scaling mask to 0-255 if necessary

        cropped_image_save_path = os.path.join(save_folder_path, f'{basename}_cropped_output.png')
        cv2.imwrite(cropped_image_save_path, masked_image)

    ########################### compute iou between generated mask and user drawing mask #######################################
    def calculate_iou(mask1, mask2):
        mask1 = (mask1 > 128).astype(np.bool_)  # Convert grayscale to binary
        mask2 = (mask2 > 128).astype(np.bool_)  # Convert grayscale to binary
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)
        return iou
    # Directory paths
    reference_mask_folder_path = mask_folder_path
    mask_folder_path = save_folder_path
    

    # Gather mask file paths
    mask_files = [f for f in os.listdir(mask_folder_path) if f.endswith('_mask_output.png')]
    reference_masks = [f for f in os.listdir(reference_mask_folder_path) if f.startswith(color) and f.endswith('.png')]

    iou_results = []

    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder_path, mask_file)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        for ref_mask in reference_masks:
            ref_mask_path = os.path.join(reference_mask_folder_path, ref_mask)
            ref_mask_image = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Calculate IoU
            iou = calculate_iou(mask_image, ref_mask_image)
            iou_results.append((mask_file, ref_mask, iou))

    # Calculate the average IoU
    average_iou = sum(iou[2] for iou in iou_results) / len(iou_results)
    # Sort IoU results
    iou_results.sort(key=lambda x: x[2], reverse=True)
    print('iou sort:' + f'{iou_results}')
    print('average IoU:' + f'{average_iou}')
    ########################### compute clip score between generated cropped image and user text input #######################################
    def calculate_clip_score(image_path, text):
        # Open the image and process it within a 'with' block to ensure it's properly closed
        with Image.open(image_path) as image:
            # Apply preprocessing and move to the device
            image = preprocess(image).unsqueeze(0).to(device)

        # Tokenize the text and move to the device
        text_tokens = clip.tokenize([text]).to(device)
        
        # Calculate similarity without gradients
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            text_features = clip_model.encode_text(text_tokens)
            similarity = torch.cosine_similarity(image_features, text_features)
        return similarity.item()
    crop_folder_path = save_folder_path
    # Gather cropped image file paths
    crop_files = [f for f in os.listdir(crop_folder_path) if f.endswith('_cropped_output.png')]

    clip_results = []

    for crop_file in crop_files:
        crop_path = os.path.join(crop_folder_path, crop_file)
        description = texts
        
        # Calculate CLIP score
        clip_score = calculate_clip_score(crop_path, description)
        clip_results.append((crop_file, clip_score))
        # clip_results.append(crop_file)
        

    # Sort CLIP results
    clip_results.sort(key=lambda x: x[1], reverse=True)
    print('clip sort:' + f'{clip_results}')

    ###################################### compute weighted top 4 for the clip score and IoU ##########################
    clip_scores = {file.split('_')[0] + '_' + file.split('_')[1]: score for file, score in clip_results}
    iou_scores = {file.split('_')[0] + '_' + file.split('_')[1]: score for file, ref,score in iou_results}

    total_scores = []
    for prefix in clip_scores.keys():
        if prefix in iou_scores:
            total_score = clip_scores[prefix] + iou_scores[prefix]
            total_scores.append((prefix, total_score))

    top_scores = sorted(total_scores, key=lambda x: x[1], reverse=True)[:4]

    for prefix, total_score in top_scores:
        print(f"Prefix: {prefix}, Total Score: {total_score}")

    ############################## only keep the top 4 image ##################################
    def rename_and_cleanup_images(image_folder_path, top_files):
        all_files = os.listdir(image_folder_path)
        top_file_prefixes = [file.split('_')[0] + '_' + file.split('_')[1] for file, _ in top_files]
        for file_name in all_files:
            if not any(file_name.startswith(prefix) for prefix in top_file_prefixes):
                file_path = os.path.join(image_folder_path, file_name)
                os.remove(file_path)
                print(f"Deleted {file_path}")

    rename_and_cleanup_images(image_folder_path, top_scores)

    return jsonify({"error": 1001, "msg": "上传失败"})

def apply_transform(image, scale_x, scale_y, trans_x, trans_y):
    if float(scale_x) != 1.0 or float(scale_y) != 1.0:
        trans_x = trans_y = 0.0

    trans_x = float(trans_x) * 2.0
    trans_y = float(trans_y) * 2.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x_min, y_min, w, h = cv2.boundingRect(c)
    else:
        x_min, y_min = 0, 0

    # Create transformation matrix for scaling and translation
    M = np.array([
        [scale_x, 0, trans_x],
        [0, scale_y, trans_y]
    ], dtype=np.float32)

    original_pt = np.array([x_min, y_min, 1], dtype=np.float32)  
    transformed_pt = M.dot(original_pt)  

    #  new_pt - (x_min, y_min)
    offset = transformed_pt - np.array([x_min, y_min], dtype=np.float32)

    # 调整平移参数，抵消偏移量
    M_adjusted = M.copy()
    M_adjusted[0, 2] -= offset[0]
    M_adjusted[1, 2] -= offset[1]

    if float(scale_x) != 1.0 or float(scale_y) != 1.0:
        # Apply the transformation to the entire image
        transformed_img = cv2.warpAffine(image, M_adjusted, (image.shape[1], image.shape[0]))
        
        # Apply the transformation to the mask as well
        transformed_mask = cv2.warpAffine(mask, M_adjusted, (mask.shape[1], mask.shape[0]))
    else:
        # Apply the transformation to the entire image
        transformed_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Apply the transformation to the mask as well
        transformed_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    
    # Create a black background image
    result_img = np.zeros_like(image)
    
    # Combine the transformed image with the transformed mask
    for c in range(3):  # Apply mask to each channel
        result_img[:, :, c] = np.where(transformed_mask > 0, transformed_img[:, :, c], 0)

    return result_img

def apply_transform_seg_mask(image, scale_x, scale_y, trans_x, trans_y):
    if float(scale_x) != 1.0 or float(scale_y) != 1.0:
        trans_x = trans_y = 0.0

    trans_x = float(trans_x) * 2.0
    trans_y = float(trans_y) * 2.0

    # Convert the image to grayscale and create a binary mask for non-white areas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)  # Invert thresholding for white background

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x_min, y_min, w, h = cv2.boundingRect(c)
    else:
        x_min, y_min = 0, 0

    # Create transformation matrix for scaling and translation
    M = np.array([
        [scale_x, 0, trans_x],
        [0, scale_y, trans_y]
    ], dtype=np.float32)

    original_pt = np.array([x_min, y_min, 1], dtype=np.float32) 
    transformed_pt = M.dot(original_pt)  #  [new_x, new_y]

    #  new_pt - (x_min, y_min)
    offset = transformed_pt - np.array([x_min, y_min], dtype=np.float32)

    M_adjusted = M.copy()
    M_adjusted[0, 2] -= offset[0]
    M_adjusted[1, 2] -= offset[1]

    if float(scale_x) != 1.0 or float(scale_y) != 1.0:
        # Apply the transformation to the entire image
        transformed_img = cv2.warpAffine(image, M_adjusted, (image.shape[1], image.shape[0]))
        
        # Apply the transformation to the mask as well
        transformed_mask = cv2.warpAffine(mask, M_adjusted, (mask.shape[1], mask.shape[0]))
    else:
        # Apply the transformation to the entire image
        transformed_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Apply the transformation to the mask as well
        transformed_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    
    # Create a white background image
    result_img = np.ones_like(image) * 255  # White background
    
    # Combine the transformed image with the transformed mask
    for c in range(3):  # Apply mask to each channel
        result_img[:, :, c] = np.where(transformed_mask > 0, transformed_img[:, :, c], 255)

    return result_img

#getMaskUpdate
@app.route('/api/test/getMaskUpdate/',methods=['GET','POST'])
def getMaskUpdate():
    mask_f = request.files["mask_image"]
    num=(int)(request.form["num"])
    transformation_list=[]
    for i in range(num):
        trans_obj={}
        trans_obj["left_move"]=request.form["left_move"+str(i)]
        trans_obj["top_move"]=request.form["top_move"+str(i)]
        trans_obj["scaleX"]=request.form["scaleX"+str(i)]
        trans_obj["scaleY"]=request.form["scaleY"+str(i)]
        # trans_obj["modified"]=True
        trans_obj["modified"]=request.form["modified"+str(i)]
        transformation_list.append(trans_obj)

    color = ['blue','yellow','green','red']
    for i in range(num):
        if transformation_list[i]["modified"]=="true":
            mask_alternate_path = r"../frontend/src/assets/mask/"

            # Add the images to the list if they exist
            if os.path.exists(fr'..\frontend\src\assets\anchor_images\{color[i]}\anchor_img_cropped_output.png'):
                image_semantic_mask = cv2.imread(fr'..\frontend\src\assets\anchor_images\{color[i]}\anchor_img_mask_output_with_color.png')
                image_cropped = cv2.imread(fr'..\frontend\src\assets\anchor_images\{color[i]}\anchor_img_cropped_output.png')
                image_mask = cv2.imread(fr'..\frontend\src\assets\anchor_images\{color[i]}\anchor_img_mask_output.png')


                scale_x=transformation_list[i]["scaleX"]
                scale_y=transformation_list[i]["scaleY"]
                trans_x=transformation_list[i]["left_move"]
                trans_y=transformation_list[i]["top_move"]
                transformed_image_semantic_mask = apply_transform_seg_mask(image_semantic_mask, scale_x, scale_y, trans_x, trans_y)
                transformed_image_mask = apply_transform(image_mask, scale_x, scale_y, trans_x, trans_y)
                transformed_image_cropped = apply_transform(image_cropped, scale_x, scale_y, trans_x, trans_y)

                cv2.imwrite(fr'..\frontend\src\assets\anchor_images\{color[i]}\anchor_img_mask_output_with_color.png', transformed_image_semantic_mask)
                cv2.imwrite(fr'..\frontend\src\assets\anchor_images\{color[i]}\anchor_img_mask_output.png', transformed_image_mask)
                cv2.imwrite(fr'..\frontend\src\assets\anchor_images\{color[i]}\anchor_img_cropped_output.png', transformed_image_cropped)


            else:
                scale_x=transformation_list[i]["scaleX"]
                scale_y=transformation_list[i]["scaleY"]
                trans_x=transformation_list[i]["left_move"]
                trans_y=transformation_list[i]["top_move"]
                alternate_mask_files = [f for f in os.listdir(mask_alternate_path) if f.startswith(color[i]) and f.endswith('.png')]
                if alternate_mask_files:
                    alternate_mask_path = os.path.join(mask_alternate_path, alternate_mask_files[0])
                    image_mask = cv2.imread(alternate_mask_path)
                    transformed_image_mask = apply_transform(image_mask, scale_x, scale_y, trans_x, trans_y)
                    cv2.imwrite(alternate_mask_path, transformed_image_mask)

    mask_img = file2img(mask_f)  
    mask_img.save(r'../frontend/src/assets/anchor_images/overall/combined_mask_output.png')
    combine_images_overall()
    return jsonify({"error": 1001, "msg": "上传失败"})