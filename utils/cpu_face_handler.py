import os
import cv2
import numpy as np
from typing import Tuple, List

class CPUFaceHandler:
    # Drop-in replacement for the mediapipe-based handler.
    # Returns bounding boxes as relative coordinates [x1,y1,x2,y2] in [0,1],
    # preserving the exact interface expected by facecrop.py.
    #
    # Detection priority:
    #   1. OpenCV DNN with ResNet-SSD  (auto-downloaded, higher accuracy)
    #   2. Haar cascade fallback        (always available in cv2)

    def __init__(self, model_selection: int = 1, min_detection_confidence: float = 0.0):
        import urllib.request

        proto_path  = "/content/SoulX-FlashHead/deploy.prototxt"
        model_path  = "/content/SoulX-FlashHead/res10_300x300_ssd_iter_140000.caffemodel"
        self.use_dnn  = False
        self.use_haar = False

        if not os.path.exists(proto_path):
            try:
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                    proto_path)
            except Exception:
                pass

        if not os.path.exists(model_path):
            try:
                urllib.request.urlretrieve(
                    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                    model_path)
            except Exception:
                pass

        if os.path.exists(proto_path) and os.path.exists(model_path):
            self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            self.use_dnn = True
        else:
            cascade_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_xml)
            self.use_haar = True

    def detect(self, image: np.ndarray) -> Tuple[List, List]:
        # Returns (bboxes, scores) with bboxes in relative [0,1] coordinates
        bboxes, scores = [], []
        img_h, img_w = image.shape[:2]

        if self.use_dnn:
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf > 0.5:
                    x1 = detections[0, 0, i, 3]
                    y1 = detections[0, 0, i, 4]
                    x2 = detections[0, 0, i, 5]
                    y2 = detections[0, 0, i, 6]
                    bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                    scores.append(conf)

        elif self.use_haar:
            gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                bboxes.append([x/img_w, y/img_h, (x+w)/img_w, (y+h)/img_h])
                scores.append(1.0)

        return bboxes, scores

    def __call__(self, image: np.ndarray) -> Tuple[List, List]:
        return self.detect(image)
