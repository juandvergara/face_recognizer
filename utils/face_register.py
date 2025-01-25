import cv2
import numpy as np
import face_recognition
import csv
import os

class FaceRegister:
    def __init__(self, name_file_to_save='known_faces'):
        print('FaceDetector initialized')
        self.name_file_to_save = name_file_to_save
        print(f'File to save: {self.name_file_to_save}.csv')

    def save_new_face(self, new_face_encoding, new_name):
        directory = "data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        path = os.path.join(directory, self.name_file_to_save + ".csv")
        print(path)
        if os.path.exists(path):
            with open(path, 'a', newline='') as csvfile:
                facewriter = csv.writer(csvfile)
                facewriter.writerow([new_name, new_face_encoding.tolist()])
        else:
            with open(path, 'w', newline='') as csvfile:
                facewriter = csv.writer(csvfile)
                facewriter.writerow(['name', 'encoding'])
                facewriter.writerow([new_name, new_face_encoding.tolist()])
        print('Face saved!')

    def register_face(self):
        video_capture = cv2.VideoCapture(0)
        name = ''
        counter_frames = 0
        total_frames = 10

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            if ret == False: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            cv2.imshow('Video', frame)

            if face_locations != []:
                if name == '': name = input('Enter the name of the person: ')
                print('Registering face...')
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                self.save_new_face(face_encodings[0], name)
                counter_frames += 1
                if counter_frames == total_frames: 
                    print('Face registered!')
                    break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
