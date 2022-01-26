from fastapi import FastAPI ,File ,UploadFile
from keras.models import load_model
from PIL import Image
from io import BytesIO
import numpy as np
import uvicorn

app = FastAPI()
model= load_model("vgg-16-model.h5")

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/categories_classification")
async def image_classifier(file: UploadFile=File(...)):
    classes = {0: 'driver_seat', 1: 'exterior_left', 2: 'exterior_right', 3: 'interior_backseat',
               4: 'interior_driver_side', 5: 'interior_passenger_side',
               6: 'tire', 7: 'negative', 8: 'odometer', 9: 'passenger_seat', 10: 'registration_card',
               11: 'verification_card'}
    exention=file.filename.split(".")[-1] in ("jpg","jpeg","png")
    if not exention:
        return {"class":'',"score":'',"model":'categories classification' ,"meta":'fail'}
    image=read_imagefile(await file.read())
    image=np.asarray(image.resize((224,224)))[..., :3]
    image=np.expand_dims(image,0)
    result = model.predict(image)[0]
    o= {"class":classes[np.argmax(result)],"score:":result[np.argmax(result)] * 100,"model":'categories classification',
        "meta":'pass'}
    return o
if __name__ =='main':
    uvicorn.run(app,host='127.0.0.1',port=8000)