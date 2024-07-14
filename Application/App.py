import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from keras.preprocessing.image import load_img
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inc_resnet_preprocess_input
from keras.applications.nasnet import NASNetLarge, preprocess_input as nasnet_preprocess_input
from keras.models import load_model, Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda
from PIL import Image, ImageTk

# Load the model
model_path = 'C:/CODE/Code/CODE ON GITHUB/Dog_Breed_Classfication/Model/DogClassification.h5'
model = load_model(model_path)

# Define the image size
img_size = (331, 331, 3)

# Define the classes (you should have this list from your training process)
classes = classes = [
    'affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
    'american_staffordshire_terrier', 'appenzeller', 'australian_terrier',
    'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog',
    'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
    'border_collie', 'border_terrier', 'borzoi', 'boston_bull',
    'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard',
    'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan',
    'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber',
    'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont',
    'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter',
    'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever',
    'french_bulldog', 'german_shepherd', 'german_short-haired_pointer',
    'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane',
    'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound',
    'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound',
    'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie',
    'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever',
    'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois',
    'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
    'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound',
    'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon',
    'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone',
    'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed',
    'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
    'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
    'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle',
    'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier',
    'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner',
    'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet',
    'wire-haired_fox_terrier', 'yorkshire_terrier'
]   # replace with the list of breed names
n_classes = len(classes)
class_to_num = dict(zip(classes, range(n_classes)))

def get_features(model_name, model_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False, input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)
    feature_maps = feature_extractor.predict(data, verbose=1)
    return feature_maps

def preprocess_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img

def extract_features(image):
    inception_features = get_features(InceptionV3, preprocess_input, img_size, image)
    xception_features = get_features(Xception, xception_preprocess_input, img_size, image)
    nasnet_features = get_features(NASNetLarge, nasnet_preprocess_input, img_size, image)
    inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocess_input, img_size, image)
    
    final_features = np.concatenate([inception_features, xception_features, nasnet_features, inc_resnet_features], axis=-1)
    return final_features

def predict_breed(image_path):
    image = preprocess_image(image_path)
    features = extract_features(image)
    prediction = model.predict(features)
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class

class DogBreedClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Dog Breed Classifier")

        self.label = Label(master, text="Select an image to predict its breed")
        self.label.pack()

        self.image_label = Label(master)
        self.image_label.pack()

        self.predict_button = Button(master, text="Select Image", command=self.select_image)
        self.predict_button.pack()

        self.result_label = Label(master, text="")
        self.result_label.pack()

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img = img.resize((200, 200), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img
            predicted_breed = predict_breed(file_path)
            self.result_label.config(text=f"Predicted Breed: {predicted_breed}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DogBreedClassifierApp(root)
    root.mainloop()
