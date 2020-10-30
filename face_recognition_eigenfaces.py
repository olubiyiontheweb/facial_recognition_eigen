import cv2
import numpy as np
import os

# This function convert images to grayscale and detect faces in images


def detect_face(input_img):
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        image, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return -1, -1
        print("No face detected")
    (x, y, w, h) = faces[0]
    return image[y:y+w, x:x+h], faces[0]


# Prepare images for training
def prepare_training_data(training_data_folder_path):
    detected_faces = []
    face_labels = []
    traning_image_dirs = os.listdir(training_data_folder_path)
    for dir_name in traning_image_dirs:
        label = int(dir_name)
        training_image_path = training_data_folder_path + "/" \
            + dir_name
        training_images_names = os.listdir(training_image_path)
        for image_name in training_images_names:
            image_path = training_image_path + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            if face is not -1:
                resized_face = cv2.resize(face, (121, 121),
                                          interpolation=cv2.INTER_AREA)
                detected_faces.append(resized_face)
                face_labels.append(label)
    return detected_faces, face_labels


# call the prepare training method and pass the training data path
detected_faces, face_labels = prepare_training_data(
    "dataset/training-data")

print("No of faces detected : ", len(detected_faces))
print("No of faces detected training folders: : ", len(face_labels))


eigenfaces_recognizer = cv2.face.EigenFaceRecognizer_create()
# Train models and Extract eigen face data (eigen vectors) from detected faces
eigenfaces_recognizer.train(detected_faces, np.array(face_labels))


def draw_rectangle(test_image, rect):
    (x, y, w, h) = rect
    cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)


def draw_text(test_image, label_text, x, y):
    cv2.putText(test_image, label_text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# predict person identified in face
def predict(test_image):
    detected_face, rect = detect_face(test_image)
    resized_test_image = cv2.resize(detected_face, (121, 121),
                                    interpolation=cv2.INTER_AREA)

    # Predict face label for the test image
    label = eigenfaces_recognizer.predict(resized_test_image)
    label_text = tags[label[0]]
    draw_rectangle(test_image, rect)
    draw_text(test_image, label_text, rect[0], rect[1]-5)
    return test_image, label_text


tags = ['Junichiro Koizumi', 'Alvaro Uribe', 'George Robertson', 'George W Bush',
        'Arnold Schwarzenegger', 'Britney Spears', 'Atal Bihari Vajpay']

test_image = cv2.imread('dataset/test-data/5/Arnold_Schwarzenegger_0021.jpg')

predicted_image, label = predict(test_image)

cv2.imshow("Predicted Image", cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB))

key = cv2.waitKey(0)
if key == ord('s'):
    cv2.imwrite("detected_person.jpg", predicted_image)
    print("Image saved")

print("Predicated Person:", label)
