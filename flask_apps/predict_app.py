import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
    global model
    model = load_model('VGG16_cats_and_dogs.h5')
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print(" * Loading Keras model...")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    
    prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'dog': prediction[0][0],
            'cat': prediction[0][1]
        }
    }
    return jsonify(response)

#@app.route("/predict", methods=["POST"])
#def predict():
#message = request.get_json(force=True)#set message = JSON of post request
#      encoded = message['image']#encoded is key from image in message variable
#decoded = base64.b64decode(encoded)#decoded is assigned decoded data
#      image = Image.open(io.BytesIO(decoded))#wrap bytes in io into image.open
#      processed_image = preprocess_image(image, target_size=(224, 224))
      # pass image to VGG as size (224, 224)
      
 #     prediction = model.predict(processed_image).tolist()
     
  #    response = {
   #       'prediction' : { #has key called prediction, which is a key
    #          'dog': prediction[0][0], #set value 0th element, 0th list
     #         'cat': prediction[0][1] #set value as 1st element, 0th list
      #    }
     # }
     # return jsonify(response)
      #1. get message that comes in
      #2. get the encoded portion from it
      #3. Decode the message
      #4. Create an image from the decoded message
      #5. Preprocess that image
      
      #6. Pass that image to our model for prediction
      #7. Send the image as JSON back to the client