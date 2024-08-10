import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
import numpy as np

face_model = keras.models.load_model('face_model')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
emoji_size = 150
mapper = {
    0: "happy",
    1: "sad",
    2: "suprise",
    3: "neutral"
}

video_capture = cv2.VideoCapture(0)

while True:
    _, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        image_data = np.array(cv2.resize(roi_gray, (48, 48))) / 255
        sample_img = image_data.reshape((1, 48, 48, 1))
        predict_class = np.argmax(face_model.predict(sample_img))
        cv2.putText(frames, mapper[predict_class],
                    (x+5, y-5), font, 1, (255, 255, 255), 2)

        emoji = cv2.imread(f"./emojis/{predict_class}.png")
        emoji = cv2.resize(emoji, (emoji_size, emoji_size))
        emoji2gray = cv2.cvtColor(emoji, cv2.COLOR_BGR2GRAY)
        _, emoji_mask = cv2.threshold(emoji2gray, 1, 255, cv2.THRESH_BINARY)
        roi_emoji = frames[-emoji_size-10:-10, -emoji_size-10:-10]
        roi_emoji[np.where(emoji_mask)] = 0
        roi_emoji += emoji


    cv2.imshow('Video', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
