from fastapi import FastAPI, UploadFile
from infer import predict_image
from PIL import Image
import io

app = FastAPI()


labels = {0: 'tench',
          1: 'English springer',
          2: 'cassette player',
          3: 'chainsaw',
          4: 'church building',
          5: 'horn',
          6: 'garbage truck',
          7: 'gas pump',
          8: 'golf ball',
          9: 'parachute'}


@app.get("/")
async def root():
    return {"message": "Welcome to the CNN"}


@app.post("/predict")
async def infer(img_file: UploadFile):

    contents = await img_file.read()
    img = Image.open(io.BytesIO(contents))
    class_label = predict_image(img)
    return {"label": labels[class_label]}
