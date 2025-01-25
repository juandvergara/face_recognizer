import cv2
import numpy as np
import face_recognition

process_this_frame = True

color = (0, 0, 255)

video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    if ret == False: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if face_locations != []:
        for face_location in face_locations:
            cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]),  color, 2)
    # Display the resulting image'''

    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break