import pandas as pd
import io
import math

from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

import numpy as np
import torch
import torch.nn.functional as F
from beheaded_inception3 import beheaded_inception_v3
from utils import as_matrix

import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image

# Create FastAPI instance
app = FastAPI()

exp_id = '0'

def get_bestModel(runs):
    best_loss = math.inf
    run_id = None
    for r in runs:
        loss = r.data.metrics['validation_loss']
        if loss < best_loss:
            best_loss = loss
            run_id = r.info.run_id
    return run_id

# Search for best run
client = MlflowClient()
runs = client.search_runs(exp_id)
best_run = get_bestModel(runs)

path = f'mlartifacts/{exp_id}/{best_run}/artifacts/ImageCaption'

print(best_run)

best_model = mlflow.pytorch.load_model(path)


inception = beheaded_inception_v3().eval()

vocab = None

with open('vocab.txt') as file:
    vocab = [line.rstrip() for line in file]

word_to_index = {w: i for i, w in enumerate(vocab)}

eos_ix = word_to_index['#END#']
unk_ix = word_to_index['#UNK#']
pad_ix = word_to_index['#PAD#']

def generate_caption(image, caption_prefix = ('#START#',), t=1, sample=True, max_len=100):
    global best_model

    best_model = best_model.cpu().eval()

    assert isinstance(image, np.ndarray) and np.max(image) <= 1\
           and np.min(image) >= 0 and image.shape[-1] == 3
    
    image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)
    
    _, vectors_neck, _ = inception(image[None])
    caption_prefix = list(caption_prefix)
    
    for _ in range(max_len):
        
        prefix_ix = as_matrix([caption_prefix], word_to_index, unk_ix, pad_ix)
        prefix_ix = torch.tensor(prefix_ix, dtype=torch.int64)
        next_word_logits = best_model.forward(vectors_neck, prefix_ix)[0, -1]
        next_word_probs = F.softmax(next_word_logits, -1).detach().numpy()
        
        assert len(next_word_probs.shape) == 1, 'probs must be one-dimensional'
        next_word_probs = next_word_probs ** t / np.sum(next_word_probs ** t) # apply temperature

        if sample:
            next_word = np.random.choice(vocab, p=next_word_probs) 
        else:
            next_word = vocab[np.argmax(next_word_probs)]

        caption_prefix.append(next_word)

        if next_word == '#END#':
            break

    return ' '.join(caption_prefix[1:-1])

# Create POST endpoint with path '/predict'
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content))
    # img = resize(img, (299, 299))
    print(img.size)
    pred_caption = generate_caption(img)
    print(f'pred_caption: {pred_caption}')
    json_compatible_item_data = jsonable_encoder(pred_caption)
    print(f'json_compatible_item_data: {json_compatible_item_data}')
    return JSONResponse(content=json_compatible_item_data)

@app.get("/")
async def main():
    content = """
    <body>
    <h2> Welcome to Image Caption</h2>
    <p> TMLflow and FastAPI instances have been set up successfully </p>
    <p> You can view the (Streamlit UI) by heading to http://localhost:8501 </p>
    </body>
    """
    return HTMLResponse(content=content)