import numpy as np
import cv2
from keras.models import load_model

model = load_model('model.h5')

class_names = ["d"]

IMG_SIZE = 100

cap = cv2.VideoCapture(0)

print("Starting live fruit identification...Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]
    label = f'{class_names[class_index]} ({confidence*100:.2f})'

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Live Fruit Identification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()