# # import numpy as np
# # # from fastapi import FastAPI, File, UploadFile
# # from fastapi import FastAPI, File, UploadFile
# # import uvicorn
# # from PIL import Image
# # from io import BytesIO
# # import tensorflow as tf
# # from fastapi.middleware.cors import CORSMiddleware
# #
# # app = FastAPI()
# # origins = [
# #     "http://localhost",
# #     "http://localhost:3000"
# # ]
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=origins,
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )
# # MODEL = tf.keras.models.load_model("../models/2")
# # CLASS_NAMES = ["Fractured", "Not Fractured"]
# #
# #
# # @app.get("/ping")
# # async def ping():
# #     return "hello,I am Alive"
# #
# #
# # def read_file_as_image(data) -> np.ndarray:
# #     image = np.array(Image.open(BytesIO(data)))
# #     return image
# #
# #
# # @app.post("/predict")
# # async def predict(
# #     file: UploadFile = File(...)
# # ):
# #     image = read_file_as_image(await file.read())
# #     img_batch = np.expand_dims(image, 0)
# #
# #     prediction = MODEL.predict(img_batch)
# #     predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
# #     confidence = np.max(prediction[0])
# #
# #     return {
# #         'class': predicted_class,
# #         'confidence': float(confidence)
# #     }
# #
# #
# # if __name__ == "__main__":
# #     uvicorn.run(app, host="localhost", port=8000)
# from PIL import Image
# import numpy as np
# from io import BytesIO
# from fastapi import FastAPI, File, UploadFile
# import tensorflow as tf
# import uvicorn
# from fastapi.middleware.cors import CORSMiddleware
#
# app = FastAPI()
# origins = [
#     "http://localhost",
#     "http://localhost:3000"
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# MODEL = tf.keras.models.load_model("../models/2")
# CLASS_NAMES = ["Fractured", "Not Fractured"]
#
# # Define target image size
# TARGET_SIZE = (224, 224)
#
# @app.get("/ping")
# async def ping():
#     return "hello, I am Alive"
#
#
# def read_file_as_image(data) -> np.ndarray:
#     # Open image from bytes
#     image = Image.open(BytesIO(data))
#     # Resize image
#     image = image.resize(TARGET_SIZE)
#     # Convert image to NumPy array
#     image_array = np.array(image)
#     return image_array
#
#
# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)
#
#     prediction = MODEL.predict(img_batch)
#     predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
#     confidence = np.max(prediction[0])
#
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL = tf.keras.models.load_model("../models/2")
CLASS_NAMES = ["Fractured", "Not Fractured"]

# Define target image size
TARGET_SIZE = (224, 224)


@app.get("/ping")
async def ping():
    return "hello, I am Alive"


def read_file_as_image(data, grayscale=True) -> np.ndarray:
    # Open image from bytes
    image = Image.open(BytesIO(data))
    # Resize image
    image = image.resize(TARGET_SIZE)

    if grayscale:
        # Convert the image to grayscale
        image = image.convert('L')

    # Convert image to NumPy array
    image_array = np.array(image)
    return image_array


@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        grayscale: bool = True  # Add a parameter to specify grayscale conversion
):
    image = read_file_as_image(await file.read(), grayscale=grayscale)
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
