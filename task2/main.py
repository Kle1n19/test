import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import spacy

model = tf.keras.models.load_model('image_classifier_32x32.h5')

animal_classes = ["SEAL", "DOLPHIN", "WHALE", "SHARK", "WOLF", "TIGER", "BEAR", "ELEPHANT", "RACOON", "SPIDER"]

ner_model = spacy.load("animal_ner_model")

def process_image(image_path):
    img = image.load_img(image_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image_class(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return animal_classes[predicted_class]

def get_user_input():
    user_input = input("What animal do you think is in the image? ")

    return user_input

def extract_animal_from_text(user_input):
    doc = ner_model(user_input)
    animals_in_text = [ent.text for ent in doc.ents if ent.label_ in animal_classes]
    return animals_in_text

def main():
    image_path = input("Please enter the path to the image: ")
    img_array = process_image(image_path)
    predicted_class = predict_image_class(img_array)
    user_input = get_user_input()
    animals_in_text = extract_animal_from_text(user_input)

    if animals_in_text:
        predicted_nlp_class = animals_in_text[0]
        return predicted_class.lower() == predicted_nlp_class.lower()
    else:
        return False

if __name__ == "__main__":
    result = main()
    print(f"Do the image and the user's answer match? {result}")
