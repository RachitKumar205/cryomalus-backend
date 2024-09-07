from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import Optional
from random import randrange
from thermal_cnn import run_inference, load_actual_coordinates  # Import inference functions
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import io
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/infer")
async def infer_coordinates():
    data_dir = "thermal_dataset_output"  # Directory where the dataset is stored
    model_path = "thermal_cnn_model.pth"  # Path to the saved model
    series_idx = randrange(93)  # Select a random series index

    # Run inference to get predicted coordinates
    predicted_coordinates = run_inference(data_dir, model_path, series_idx=series_idx)

    # Load actual coordinates from metadata
    actual_coordinates = load_actual_coordinates(data_dir, series_idx)

    # Generate heatmap from the CSV data (for demonstration, assuming grid size 100x100)
    series_images = []
    for t in range(5):  # Assuming 5 time steps
        img_path = f"{data_dir}/series_{series_idx:04d}/thermal_data_step_{t:03d}.csv"
        df = pd.read_csv(img_path)
        series_images.append(df.to_numpy())

    # Create an averaged heatmap from the series images
    heatmap_data = np.mean(series_images, axis=0)

    # Plot heatmap
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    plt.colorbar()

    # Save image to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()

    # Send both the predicted coordinates and the heatmap image
    return JSONResponse(content={
        "predicted_coordinates": predicted_coordinates.tolist(),
        "actual_coordinates": actual_coordinates.tolist(),
        "heatmap_image": "heatmap.png"
    })

@app.get("/heatmap")
async def get_heatmap():
    img_buf = io.BytesIO()
    # Generate heatmap again if needed
    # Alternatively, save the heatmap from the infer_coordinates function and serve that file

    # Assuming some sample heatmap generation
    plt.imshow(np.random.rand(10, 10), cmap='hot')
    plt.colorbar()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()

    return Response(img_buf.getvalue(), media_type="image/png")
