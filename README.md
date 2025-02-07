# SketchFlex

SketchFlex is designed for novice users to better utilize Image Generative Models to create their own work with enhanced control. It incorporates the following features:

1. Recommend detailed prompts using MLLM based on your initial input and sketch for better generation results.
2. Generate objects separately for you to select as shape candidates.
3. Based on your selected shapes, you can adjust the location and size of each shape, and then input them to ControlNet to generate an image based on those shapes.

## How to Use

### 1. Prepare Backend Environment

```
git clone https://github.com/SellLin/SketchFlex


conda create -n sketchflex python=3.10
conda activate sketchflex

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

cd backend
pip install requirements.txt
set FLASK_APP = app.py
flask run
```

### 2. webui API usage
we call stable diffusion webui API to use controlnet and regional prompter.
if you haven't install webui or haven't use controlnet\regional prompter extension:

-download webui: https://github.com/AUTOMATIC1111/stable-diffusion-webui<br>
-regional prompter extension: https://github.com/hako-mikan/sd-webui-regional-prompter<br>
-controlnet extension: https://github.com/Mikubill/sd-webui-controlnet

set your server url in line 341 in app.py, open webui with API Mode.

### 3. Down load models
download GoogleNews-vectors-negative300.bin : https://huggingface.co/NathaNn1111/word2vec-google-news-negative-300-bin/tree/main and put in backend\model.

download colorfulxlLightning_v16.safetensors：https://huggingface.co/recoilme/workspace/blob/main/ColorfulXL_v16.safetensors (or other basemodel you would like to use in the final generation) in backend\model.
if use other model name, change line 540 in app.py

download fastsam :https://github.com/CASIA-IVA-Lab/FastSAM and put in backend\model\mobile_sam_test\FastSAM-x.pt.
if use other fastsam version, change line 97 in app.py

### 4. frontend
```
cd frontend
```
For the first time running:
```
npm install 
```
Run the frontend and enter the local link
```
npm run dev
```
