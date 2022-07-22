from flask import Flask, render_template, request, jsonify
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = ""

# this is dict
ref ={0: 'Apple___Apple_scab',
 1: 'Apple___Black_rot',
 2: 'Apple___Cedar_apple_rust',
 3: 'Apple___healthy',
 4: 'Blueberry___healthy',
 5: 'Cherry_(including_sour)___Powdery_mildew',
 6: 'Cherry_(including_sour)___healthy',
 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 8: 'Corn_(maize)___Common_rust_',
 9: 'Corn_(maize)___Northern_Leaf_Blight',
 10: 'Corn_(maize)___healthy',
 11: 'Grape___Black_rot',
 12: 'Grape___Esca_(Black_Measles)',
 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 14: 'Grape___healthy',
 15: 'Orange___Haunglongbing_(Citrus_greening)',
 16: 'Peach___Bacterial_spot',
 17: 'Peach___healthy',
 18: 'Pepper,_bell___Bacterial_spot',
 19: 'Pepper,_bell___healthy',
 20: 'Potato___Early_blight',
 21: 'Potato___Late_blight',
 22: 'Potato___healthy',
 23: 'Raspberry___healthy',
 24: 'Soybean___healthy',
 25: 'Squash___Powdery_mildew',
 26: 'Strawberry___Leaf_scorch',
 27: 'Strawberry___healthy',
 28: 'Tomato___Bacterial_spot',
 29: 'Tomato___Early_blight',
 30: 'Tomato___Late_blight',
 31: 'Tomato___Leaf_Mold',
 32: 'Tomato___Septoria_leaf_spot',
 33: 'Tomato___Spider_mites Two-spotted_spider_mite',
 34: 'Tomato___Target_Spot',
 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 36: 'Tomato___Tomato_mosaic_virus',
 37: 'Tomato___healthy'}

def prediction(path):
    img = Image.open(path)
    img = img.resize((256,256),Image.ANTIALIAS)
    img = np.asarray(img)
    im = preprocess_input(img)
    img = np.expand_dims(im,axis = 0)
    pred = np.argmax(model.predict(img))
    print(f'The image belongs to {ref[pred]}')
    return ref[pred]

@app.route('/perdict', methods=['POST'])
def predict():
    # print("sss",request.files["myFile"])
    file1 = request.files['myFile']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
    file1.save(path)
    return jsonify({"plant": prediction(path).split("___")[0], "disease": prediction(path).split("___")[1]})

@app.route('/')
def hello_world():
	return render_template("index.html")

if __name__ == '__main__':
    model = load_model("./model/best_model.h5")
    app.run()
