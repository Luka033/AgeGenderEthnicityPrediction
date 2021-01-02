import keras
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

print(keras.__version__)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gender_classifier = load_model('gender-pred.h5')
age_classifier = load_model('age-pred.h5')
ethnicity_classifier = load_model('ethnicity-pred.h5')

class_labels = ['Male', 'Female']
ethnicity_class = ['White', 'Black', 'Asian', 'Indian', 'Other']

cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class
            preds = gender_classifier.predict(roi)[0]
            age = age_classifier.predict(roi)[0]
            ethnicity = ethnicity_classifier(roi)[0]
            label = class_labels[round(preds[0])] + " " + str(round(age[0])) + " " + ethnicity_class[preds.argmax()]

            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()