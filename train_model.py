
import numpy as np 
import cv2
from utills.preprocess import preprocess_image
from utills.database import collection
from utills.cnn_model import build_model


X = []
y = []

persons = collection.find()

for person in persons:
    label = person['label']

    for img_binary in person['images']:
        img = preprocess_image(img_binary)
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

num_classes = len(np.unique(y))
model = build_model(num_classes)

model.fit(
        X,
        y,
        epochs = 10,
        batch_size = 8
)


model.save("model/face_model.h5")
print("Model Trained Sucessfully")
