from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_pipeline

app = FastAPI()

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    # Read the uploaded file as bytes
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    output_img = predict_pipeline(img)
    
    output_img.save("output_prediction.png")
    
    buffered = io.BytesIO()
    output_img.save(buffered, format="PNG")
    encoded_output = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return as JSON
    return JSONResponse(content={"image": encoded_output})