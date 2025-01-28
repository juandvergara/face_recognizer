import csv
import numpy as np
import face_recognition
import cv2

from utils.face_register import FaceRegister
from utils.face_matcher import FaceMatcher

face_register = FaceRegister('testing_data')
face_matcher = FaceMatcher('testing_data')

def main():
    face_register.register_face()

    print("Waiting for camera...")

    video_capture = cv2.VideoCapture(0)
    while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            if ret == False: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_locations = face_recognition.face_locations(rgb_frame)
            actual_face_encodings = face_recognition.face_encodings(rgb_frame, faces_locations)

            for (top, right, bottom, left), actual_face_encoding in zip(faces_locations, actual_face_encodings):
                name = face_matcher.compare_faces(actual_face_encoding)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Video', frame)            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    


main()

'''
known_face_encodings = []
known_face_names = []
with open('known_faces.csv', 'r') as csvfile:
    facereader = csv.reader(csvfile)
    next(facereader)  # Skip the header row
    for row in facereader:
        name = row[0]
        encoding = np.array(eval(row[1]))
        known_face_names.append(name)
        known_face_encodings.append(encoding)
'''