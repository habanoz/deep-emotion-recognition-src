import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('opencv-haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('opencv-haarcascades/haarcascade_eye.xml')

faces_image_path = "/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Images/Val/Surprise/004524480-0001.jpg"

def detect_landmarks(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #output list
    face_landmark_tuples=[]

    faces = face_cascade.detectMultiScale(gray, 1.3, 2)

    print("Number of faces detected: {}".format(len(faces)))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        print("Number of eyes detected: {}".format(len(eyes)))

        face_landmark_tuples.append(((x, y, w, h), eyes))

    return face_landmark_tuples

def draw_landmarks(image,face_landmark_tuples):


    # copy original image, do not touch it
    out_image=image.copy()

    # draw for each face
    for landmart_tuple in face_landmark_tuples:
        # un-box parts from tuple
        (x, y, w, h),eyes=landmart_tuple

        # draw face rectangle
        cv2.rectangle(out_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_color = out_image[y:y + h, x:x + w]

        # draw eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return out_image



def main():
    image = cv2.imread(faces_image_path)


    # detect faces and landmarks
    face_landmark_tuples = detect_landmarks(image=image)
    image=draw_landmarks(image,face_landmark_tuples)

    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ =='__main__':
    main()