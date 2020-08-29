import face_recognition
import cv2
import numpy as np
import os
import glob
# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

"""img_dir = "images"
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)
data = []
masked_faces_encodings = []
for fl in files:
    data.append(fl)
    masked_faces_images = face_recognition.load_image_file(fl)
    masked_faces_encoding = face_recognition.face_encodings(masked_faces_images)

    masked_faces_encodings.append(masked_faces_encoding)
    masked_faces = ["Masked"]

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
print(masked_faces_encodings)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        # matches = face_recognition.compare_faces(masked_faces_encodings, face_encoding)
        name = "Unmasked"
        if name == "Unmasked":
            print("ALERT!!!!", "\a")

        # # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = masked_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(masked_faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = masked_face_names[best_match_index]

        face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


print("IMG DATA: ", data)"""








# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("face.jpeg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("face2.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # tolerance=0.0
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unmasked"
            if name == "Unmasked":
                print("ALERT!!!!", "\a")

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()










# import face_recognition
# import cv2
# import numpy as np
#
# """Does this only need to trigger when it sees a face? Otherwise just keep looping through frames until a face a found.
# Because the facial recognition is not able to recognized masked faces"""
#
# video_capture = cv2.VideoCapture(0)
#
# # Initialize some variables
# face_locations = []
# face_encodings = []
# face_names = []
# process_this_frame = True
#
# while True:
#     # Grab a single frame of video
#     ret, frame = video_capture.read()
#
#     # Resize frame of video to 1/4 size for faster face recognition processing
#     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#
#     # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#     rgb_small_frame = small_frame[:, :, ::-1]
#
#     # Only process every other frame of video to save time
#     if process_this_frame:
#         # Find all the faces and face encodings in the current frame of video
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
#
#         face_names = []
#         for face_encoding in face_encodings:
#             name = "Unmasked"
#             if name == "Unmasked":
#                 print("ALERT!!!", '\a')
#
#     process_this_frame = not process_this_frame
#
#
#     # Display the results
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         # Scale back up face locations since the frame we detected in was scaled to 1/4 size
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4
#
#         # Draw a box around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#
#         # Draw a label with a name below the face
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
#
#     # Display the resulting image
#     cv2.imshow('Video', frame)
#
#     # Hit 'q' on the keyboard to quit!
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()
#
#
#
#
#
#
#
#
#
#
# # from PIL import Image
# # import face_recognition
# # import cv2
# # import sys
#
# # faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# #
# # video_capture = cv2.VideoCapture(0)
# #
# # while True:
# #     ret, frame = video_capture.read()
# #
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     faces = faceCascade.detectMultiScale(
# #         gray,
# #         scaleFactor=1.5,
# #         minNeighbors=5,
# #         minSize=(30, 30),
# #         flags=cv2.CASCADE_SCALE_IMAGE
# #     )
# #
# #     for (x, y, w, h) in faces:
# #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0 ,255, 0), 2)
# #
# #     cv2.imshow('FaceDetections', frame)
# #
# #     if k%256 == 27:
# #         break
# #
# #     elif k%256 -- 32:
# #         img_name = "facedetect_webcam_{}.png".format(img_counter)
# #         cv2.imwrite(img_name, frame)
# #         print("{} written!".format(img_name))
# #         img_counter += 1
# #
# #     video_capture.release()
# #     cv2.destroyAllWindows()
#
#
# #
# # # cascPath = sys.argv[1]
# # faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# #
# # video_capture = cv2.VideoCapture(0)
# #
# # while True:
# #     # Capture frame-by-frame
# #     ret, frame = video_capture.read()
# #
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #
# #     faces = faceCascade.detectMultiScale(
# #         gray,
# #         scaleFactor=1.1,
# #         minNeighbors=5,
# #         minSize=(30, 30),
# #         # flags=cv2.cv2.CV_HAAR_SCALE_IMAGE
# #     )
# #
# #     # Draw a rectangle around the faces
# #     for (x, y, w, h) in faces:
# #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
# #
# #     # Display the resulting frame
# #     cv2.imshow('Video', frame)
# #
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# #
# # # When everything is done, release the capture
# # video_capture.release()
# # cv2.destroyAllWindows()
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # masked_faces = face_recognition
# #
# # known_image = face_recognition.load_image_file("mask.jpeg")
# # unknown_image = face_recognition.load_image_file("face.jpeg")
# #
# # try:
# #     known_image_encoding = face_recognition.face_encodings(known_image)[0]
# #     unknown_image_encoding = face_recognition.face_encodings(unknown_image)[0]
# # except IndexError:
# #     print("I was not able to locate any faces in at least one of the images. Check the image files. Aborting...")
# #     quit()
# #
# # known_faces = [
# #     known_image_encoding
# # ]
# #
# # results = face_recognition.compare_faces(known_faces, unknown_image_encoding)
# #
# # print("Is the unknown face face.jpg {}".format(results[0]))
# # print("Is the unknown face a new person that we have never seen before? {}".format(not True in results))
# #
#
# #
# #
# # # def face_rec():
# # #     known_image = face_recognition.load_image_file("face.jpg")
# # #     unknown_image = face_recognition.load_image_file("face.jpeg")
# # #
# # #     known_encoding = face_recognition.face_encodings(known_image)[0]
# # #     unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
# # #
# # #     results = face_recognition.compare_faces([known_encoding], unknown_encoding)
# # #     print(results)
# # #     return results
# #
# #
# #
# #
# # image = face_recognition.load_image_file("group.jpg")
# # face_locations = face_recognition.face_locations(image)
# # print("I found {} face(s) in this photograth.".format(len(face_locations)))
# # for face_location in face_locations:
# #     top, right, bottom, left = face_location
# #     print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
# #     face_image = image[top:bottom, left:right]
# #     pil_image = Image.fromarray(face_image)
# #     pil_image.show()
