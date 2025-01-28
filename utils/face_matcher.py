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
        self._known_face_encodings = []
        self._known_face_names = []

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
                    self._known_face_names.append(name)
                    self._known_face_encodings.append(encoding)
        else:
            print('No faces saved')
    
    def compare_faces(self, face_to_compare, tolerance=0.4):
        '''
        Compare faces with database\\

        :param face_to_compare: list # The image with face to compare
        :param tolerance: float # Tolerance to compare faces
        '''
        name = "Unkwown"
        self.__load_known_faces()
        comparisions = face_recognition.compare_faces(self._known_face_encodings, face_to_compare, tolerance)

        if True in comparisions:
                    first_match_index = comparisions.index(True)
                    name = self._known_face_names[first_match_index]
                    
        return name
        # TODO: Implement the face comparison
