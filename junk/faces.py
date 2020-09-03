import face_recognition
import cv2
import numpy as np
import os
import glob
import beepy

video_capture = cv2.VideoCapture(0)
# img_list = ["face.jpeg", "face2.jpg", "face3.jpg"]
known_faces_encodings = []

def image_list():
    img_dir = "images/dataset/with_mask"
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    # data = []
    img_list = []
    for file in files:
        img_list.append(file)
    # print(img_list)
    # for img in img_list:
        # face_locations = face_recognition.face_locations(img)
        # print(img)
        # if face_locations == False:
        #     img_list.pop(img)



    for img in img_list:
        known_faces = face_recognition.load_image_file(img)
        if face_recognition.face_encodings(known_faces) == []:
            pass
        else:
            known_faces_encoding = face_recognition.face_encodings(known_faces)[0]
            known_faces_encodings.append(known_faces_encoding)
    print(known_faces_encodings)


# def face_detaction():
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = []

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame,(0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:

            face_locations = face_recognition.face_locations(rgb_small_frame)
            # if face_locations:
                # print("TRUE")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            # face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

            if face_encodings == []:
                pass
            else:
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
                    if True in matches:
                        first_match_index = matches.index(True)
                        print("MATCH", "\a")
                    else:
                        print("No match", "\m")


            # print(face_landmarks_list)
            # if len(face_landmarks_list) == 0:
            #        pass
            # elif 'top_lip' in face_landmarks_list[0]:
            #     # print(face_landmarks_list[0]['chin'])
            #     print("TOP LIP", face_landmarks_list[0]['top_lip'][5][0])
            # else:
            #     print("Not here")

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) &0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

image_list()


# def face_matches():
#     for face_encoding in face_encodings:
#         matches = face_recognition.compare_faces(known)
#         if True in matches...

# face_image = face_recognition.load_image_file("face3.jpg")
# face_landmarks_list = face_recognition.face_landmarks(face_image)
# face_image_encoding = face_recognition.face_encodings(face_image)[0]

# print(face_landmarks_list)
# if 'top_lip' in face_landmarks_list[0]:
#     # print(face_landmarks_list[0]['chin'])
#     print("HERE")
# else:
#     print("Not here")
