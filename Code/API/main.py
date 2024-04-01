# import numpy as np
# # from fastapi import FastAPI, File, UploadFile
# from fastapi import FastAPI, File, UploadFile
# import uvicorn
# from PIL import Image
# from io import BytesIO
# import tensorflow as tf
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
#
# @app.get("/ping")
# async def ping():
#     return "hello,I am Alive"
#
#
# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image
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





#new Code
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

MODEL = tf.keras.models.load_model("../models/7")

TARGET_SIZE = (256, 256)

CLASS_NAMES = ["Fractured", "Not Fractured"]

@app.get("/ping")
async def ping():
    return "Hello, I am Alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize(TARGET_SIZE)
    image = image.convert('L')
    image_array = np.array(image)
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_content = await file.read()

    image = read_file_as_image(file_content)
    img_batch = np.expand_dims(image, axis=0)
    img_batch = np.expand_dims(img_batch, axis=-1)

    print(image.shape)
    print(img_batch.shape)

    if is_xray_image(file_content):
        prediction = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))

        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }

    else:
        return {
            'class': "Not an X-ray Image",
            'confidence': "null"
        }


def is_xray_image(file_content):

    xray_magic_bytes = {
        b'\x44\x49\x43\x4D': 'DICOM',
        b'\xFF\xD8\xFF': 'JPEG',
        b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A': 'PNG',
        b'\xFF\xD8\xFF\xE0': 'JPG',
    }

    header = file_content[:4]

    for magic in xray_magic_bytes.keys():
        if header.startswith(magic):
            return True

    return False


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

#REAL RUNNING CODE
# from PIL import Image
# import numpy as np
# from io import BytesIO
# import cv2
# from fastapi import FastAPI, File, UploadFile
# import tensorflow as tf
# import uvicorn
# from fastapi.middleware.cors import CORSMiddleware
#
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
# # def is_xray_image(image_array):
# #     # Convert image to grayscale if it's not already
# #     if len(image_array.shape) == 3:
# #         gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
# #     else:
# #         gray_image = image_array
# #
# #     # Calculate mean intensity
# #     mean_intensity = np.mean(gray_image)
# #
# #     # Example: Adjust the threshold based on your dataset
# #     threshold = 150
# #     return mean_intensity < threshold
# def read_file_as_image(data) -> np.ndarray:
#     # Open image from bytes
#     image = Image.open(BytesIO(data))
#     # Resize image
#     image = image.resize(TARGET_SIZE)
#     image = image.convert('L')
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
#
#
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

#
# from PIL import Image
# import numpy as np
# from io import BytesIO
# import cv2
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
#
# @app.get("/ping")
# async def ping():
#     return "hello, I am Alive"
#
#
# INTENSITY_THRESHOLD = 10000  # Example intensity threshold
# EDGE_DENSITY_THRESHOLD = 50
#
#
# # Function to check if the image is an X-ray image
# def is_xray_image(image: np.ndarray) -> bool:
#     # Check if the image has only one channel (grayscale)
#     if len(image.shape) != 2:
#         return False
#
#     # Intensity Distribution Analysis
#     # Calculate histogram of pixel intensities
#     hist = cv2.calcHist([image], [0], None, [256], [0, 256])
#
#     # Check if the histogram distribution resembles that of an X-ray image
#     # You might need to define specific thresholds based on X-ray image characteristics
#     if hist[0] > INTENSITY_THRESHOLD:
#         return False  # Not an X-ray image
#
#     # Edge Detection
#     # Apply Canny edge detection
#     edges = cv2.Canny(image, threshold1=100, threshold2=200)  # You might need to adjust the thresholds
#
#     # Check if the edge density or presence of edges resembles that of an X-ray image
#     if np.mean(edges) < EDGE_DENSITY_THRESHOLD:
#         return False  # Not an X-ray image
#
#     # If the image passes all checks, consider it as an X-ray image
#     return True
#
#
# def read_file_as_image(data) -> np.ndarray:
#     # Open image from bytes
#     image = Image.open(BytesIO(data))
#     # Resize image
#     image = image.convert('L')
#     image = image.resize(TARGET_SIZE)
#     # Convert image to NumPy array
#     image_array = np.array(image)
#     return image_array
#
#
# @app.post("/predict")
# async def predict(
#         file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#
#     # Check if the image is an X-ray image
#     if not is_xray_image(image):
#         return {
#             'error': 'The provided image is not an X-ray image.'
#         }
#
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

# from PIL import Image
# import numpy as np
# from io import BytesIO
# from fastapi import FastAPI, File, UploadFile, HTTPException
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
# def is_xray_image(image_array):
#     # Perform some basic checks to identify X-ray images
#     # Example: Check if the image has a dominant color or pattern typically found in X-rays
#     # Note: This is a basic example and may not cover all cases
#     # You may need to adjust these checks based on the characteristics of your X-ray images
#     unique_colors = np.unique(image_array)
#     if len(unique_colors) <= 5:
#         return True
#     return False

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
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # Validate that the uploaded file is an X-ray image
#     image = read_file_as_image(await file.read())
#     if not is_xray_image(image):
#         raise HTTPException(status_code=400, detail="Uploaded file is not a valid X-ray image.")
#
#     img_batch = np.expand_dims(image, 0)
#
#     prediction = MODEL.predict(img_batch)
#     predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
#     confidence = float(np.max(prediction[0]))
#
#     return {
#         'class': predicted_class,
#         'confidence': confidence
#     }
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)


