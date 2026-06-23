import os
import cv2
import numpy as np


#Create folders for dataset if they don't exist and put the images in the folders in the dataset directory. The folders should be named according to the labels defined in the LABELS dictionary.
DATASET_PATH = 'dataset'
LABELS = {
    0: 'Me',
    1: 'Sister',
    2: 'Mother',
}
IMAGE_SIZE = (200, 200)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def prepare_training_data(dataset_path):
    faces = []
    labels = []

    for label_id, name in LABELS.items():
        person_folder = os.path.join(dataset_path, name)
        if not os.path.isdir(person_folder):
            raise RuntimeError(
                f"Missing dataset folder: {person_folder}. Create folders named {', '.join(LABELS.values())} inside '{dataset_path}'."
            )

        for filename in os.listdir(person_folder):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image_path = os.path.join(person_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(detected) == 0:
                continue

            x, y, w, h = detected[0]
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, IMAGE_SIZE)
            faces.append(face)
            labels.append(label_id)

    return faces, np.array(labels, dtype=np.int32)


def build_recognizer(faces, labels):
    if not hasattr(cv2, 'face'):
        raise RuntimeError('opencv-contrib-python is required for face recognition. Install it with pip.')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)
    return recognizer


def main():
    faces, labels = prepare_training_data(DATASET_PATH)
    recognizer = build_recognizer(faces, labels)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in detected_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, IMAGE_SIZE)

            label_id, confidence = recognizer.predict(face_roi)
            name = LABELS.get(label_id, 'Unknown')
            cv2.putText(frame, f'{name}', (x, max(y - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.putText(frame, f'{confidence:.0f}', (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.3, minNeighbors=5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(roi_color, 'Eye', (ex, max(ey - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('face recognition', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()