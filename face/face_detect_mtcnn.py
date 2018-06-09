import cv2
import mxnet as mx
from mxnet_mtcnn.mtcnn_detector import MtcnnDetector

faces_image_path = "/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Images/Val/Surprise/010301054-0001.jpg"
faces_image_path = "/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Images/Val/Surprise/004524480-0001.jpg"
faces_image_path = "/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Images/Val/Angry/000149120-0001.jpg"
faces_image_path = "/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Images/Val/Angry/000149120-0057.jpg"
faces_image_path = "/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Images/Val/Angry/000149120-0056.jpg"
faces_image_path = "/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Images/Val/Happy/001537920-0010.jpg"
faces_image_path = "/mnt/sda2/dev_root/dataset/AFEW-Processed/AFEW_Images/Val/Surprise/004406878-0001.jpg"

detector = MtcnnDetector(model_folder='mxnet_mtcnn/model', ctx=mx.cpu(0), num_worker=4, accurate_landmark=False)

img = cv2.imread(faces_image_path)

# run detector
results = detector.detect_face(img)

if results is not None:

    total_boxes = results[0]
    points = results[1]

    # extract aligned face chips
    chips = detector.extract_image_chips(img, points, 144, 0)
    for i, chip in enumerate(chips):
        cv2.imshow('chip_' + str(i), chip)
        cv2.imwrite('chip_' + str(i) + '.png', chip)

    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

    for p in points:
        for i in range(5):
            cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

    cv2.imshow("detection result", draw)
cv2.waitKey(0)