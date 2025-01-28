import cv2
import numpy as np
import face_recognition
import csv
import os

class FaceRegister:
    def __init__(self, name_file_to_save='known_faces', directory='data'):
        print('FaceRegister initialized')
        self.directory = directory
        self.name_file_to_save = name_file_to_save
        print(f'File to save: {self.name_file_to_save}.csv into {self.directory}')

    def __save_new_face(self, new_face_encoding, new_name):

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        path = os.path.join(self.directory, self.name_file_to_save + ".csv")
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
    
    def register_face(self, total_frames=10, name=''):
        '''
        Register a new face into the data file, this method will open \\
        a webcam and will capture the face of the person to register

        :param total_frames: int # Number of frames to capture
        :param name: str # Name of the person to register
        '''
        self.name = name
        counter_frames = 0
        self.total_frames = total_frames

        video_capture = cv2.VideoCapture(0)
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            if ret == False: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            cv2.imshow('Video', frame)

            if face_locations != []:
                if self.name == '': self.name = input('Enter the name of the person: ')
                print('Registering face...')
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                self.__save_new_face(face_encodings[0], self.name)
                counter_frames += 1
                if counter_frames == self.total_frames: 
                    print('Face registered!')
                    break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
