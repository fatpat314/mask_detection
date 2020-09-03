import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('face_detector.xml')

# Read the input image
img = cv2.imread('not_face.jpg')

# Detect faces
faces = face_cascade.detectMultiScale(cv2.imread('/Users/patrickkelly/Desktop/2020/Projects_2020/SUP2.2/face.jpeg'), 1.1, 4)

# Draw rectangle around teh faces
for(x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

# Export the result
cv2.imwrite("face_test.jpg", img)

print('Successfully saved')
