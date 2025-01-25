import cv2
import numpy as np
import face_recognition
import csv
import os

class FaceMatcher:
    def __init__(self, name_file_to_save='known_faces', directory='data'):
        print('FaceMatcher initialized')
        self.directory = directory
        self.name_file_to_save = name_file_to_save
        self.known_face_encodings = []
        self.known_face_names = []

    def __load_known_faces(self):
        '''
        Register a new face into the data file, this method will open \\
        a webcam and will capture the face of the person to register

        :param total_frames: int # Number of frames to capture
        :param name: str # Name of the person to register
        '''
        
        path = os.path.join(self.directory, self.name_file_to_save + ".csv")
        if os.path.exists(path):
            with open(path, 'r') as csvfile:
                facereader = csv.reader(csvfile)
                next(facereader)  # Skip the header row
                for row in facereader:
                    name = row[0]
                    encoding = np.array(eval(row[1]))
                    self.known_face_names.append(name)
                    self.known_face_encodings.append(encoding)
        else:
            print('No faces saved')
    
    def compare_faces(self, face_to_compare):
        '''
        Register a new face into the data file, this method will open \\
        a webcam and will capture the face of the person to register

        :param total_frames: int # Number of frames to capture
        :param name: str # Name of the person to register
        '''
        self.__load_known_faces()
        
        # TODO: Implement the face comparison
