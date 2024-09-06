#face_recognition.py
import subprocess
import cv2
import dlib
import numpy as np
import pandas as pd
import time
import logging
import os

# Dlib face detector and models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

class FaceRecognizer:
    def __init__(self):
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.app_started = False

    def load_database(self):
        if not os.path.exists("data/features_all.csv"):
            logging.warning("'features_all.csv' not found! Run required scripts.")
            return False

        path_features_known_csv = "data/features_all.csv"
        csv_rd = pd.read_csv(path_features_known_csv, header=None)
        
        for i in range(csv_rd.shape[0]):
            features_someone_arr = []
            self.face_name_known_list.append(csv_rd.iloc[i][0])
            for j in range(1, 129):
                if csv_rd.iloc[i][j] == '':
                    features_someone_arr.append('0')
                else:
                    features_someone_arr.append(csv_rd.iloc[i][j])
            self.face_features_known_list.append(features_someone_arr)
        
        logging.info("Faces in Database: %d", len(self.face_features_known_list))
        return True

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def recognize_faces_in_image(self, img_rd):
        faces = detector(img_rd, 0)
        face_names = []
        face_coords = []

        for face in faces:
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()

            # Draw rectangle around face
            cv2.rectangle(img_rd, (left, top), (right, bottom), (0, 255, 0), 2)

            shape = predictor(img_rd, face)
            face_feature = face_reco_model.compute_face_descriptor(img_rd, shape)

            min_distance = float("inf")
            identity = "unknown"

            for idx, known_face_feature in enumerate(self.face_features_known_list):
                distance = self.return_euclidean_distance(face_feature, known_face_feature)
                if distance < min_distance:
                    min_distance = distance
                    identity = self.face_name_known_list[idx]

            if min_distance < 0.4:
                face_names.append(identity)
            else:
                face_names.append("unknown")

            face_coords.append((left, top, right, bottom))

        return face_names, face_coords

    def get_camera_stream(self):
        address = "http://192.168.0.102:8080/video"
        return cv2.VideoCapture(address)

def main():
    fr = FaceRecognizer()

    logging.basicConfig(level=logging.INFO)
    logging.info("Face Recognizer")
    
    if not fr.load_database():
        return

    stream = fr.get_camera_stream()

    # Variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    while stream.isOpened():
        ret, img_rd = stream.read()

        if not ret:
            break

        frame_count += 1

        # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Avoid division by zero
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        else:
            fps = 0
        
        # Recognize faces and draw rectangles
        face_names, face_coords = fr.recognize_faces_in_image(img_rd)

        # Display FPS and number of faces
        cv2.putText(img_rd, f"FPS: {round(fps, 2)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(img_rd, f"Number of faces: {len(face_names)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(img_rd, "Q: QUIT", (10, img_rd.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for name, (left, top, right, bottom) in zip(face_names, face_coords):
            # Display recognized name next to the face rectangle
            cv2.putText(img_rd, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if name != "unknown" and not fr.app_started:
                subprocess.Popen(["python", "app.py"])
                fr.app_started = True  # Set flag to True after starting app.py

        cv2.imshow("camera", img_rd)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
