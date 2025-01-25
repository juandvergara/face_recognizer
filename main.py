import csv
import numpy as np

from utils.face_register import FaceRegister

face_register = FaceRegister('testing_data')

def main():
    face_register.register_face()

main()


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
