import os
import pickle

import cv2
import numpy as np
import streamlit as st
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from mtcnn import MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# actors = sorted(os.listdir("dataset"))
# filenames = []
# for actor in actors:
#     for file in os.listdir(os.path.join("dataset", actor)):
#         filenames.append(os.path.join("dataset", actor, file))

# pickle.dump(filenames, open("filenames.pkl", "wb"))
# print(f"Total Files: {len(filenames)}\n")

detector = MTCNN()
filenames = pickle.load(open("filenames.pkl", "rb"))
feature_list = np.array(
    pickle.load(
        open(
            "embedding.pkl",
            "rb",
        )
    )
)
model = VGGFace(
    model="resnet50",
    include_top=False,
    input_shape=(224, 224, 3),
    pooling="avg",
)


def save_uploaded_image(uploaded_img):
    try:
        with open(
            os.path.join("uploads", str(uploaded_img.name)), "wb"
        ) as f:
            f.write(uploaded_img.getbuffer())
        return True
    except Exception as err:
        print(f"[ERROR]: {err}")
        return False


def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    x, y, w, h = results[0]["box"]
    face = img[y : y + h, x : x + w]

    # extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    face_array = face_array.astype("float32")

    # expanded_img = np.expand_dim(face_array, axis=0)
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


def recommend(feature_list, feature):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(
            cosine_similarity(
                feature.reshape(1, -1), feature_list[i].reshape(1, -1)
            )[0][0]
        )
    index_pos = sorted(
        list(enumerate(similarity)), reverse=True, key=lambda x: x[1]
    )[0][0]
    return index_pos


st.title("Which bollywood celebrity are you?")
uploaded_image = st.file_uploader("Choose an image")
if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)
        
        # extract the features
        features = extract_features(
            os.path.join("uploads", uploaded_image.name),
            model,
            detector,
        )
        
        # recommend
        index_pos = recommend(feature_list, features)
        predicted_actor = " ".join(
            filenames[index_pos].split("\\")[1].split("_")
        )
        
        # display
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Your uploaded image")
            st.image(uploaded_image)
        with col2:
            st.subheader("Seems like " + predicted_actor)
            st.image(filenames[index_pos], width=300)
