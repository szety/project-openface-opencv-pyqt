# -*- coding: utf-8 -*-
"""
Face recognition using opencv and openface model
in a QT applicaiton.
"""

# Author: Tom Sze <sze.takyu@gmail.com>

import os
from os import walk
import sys
from PyQt5.QtCore import QMutex, QWaitCondition, pyqtSignal, \
    QThread, pyqtSlot, Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, \
    QPushButton, QSizePolicy, QLineEdit, QLabel, \
    QApplication, QFileDialog
import imutils
import cv2
import numpy as np
from face_class.face_recognizer import FaceRecognizer


class MyWidget(QWidget):

    def __init__(self):
        super(MyWidget, self).__init__()

        self.cur_path = os.path.split(os.path.realpath(__file__))[0]
        self.im_original = None
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.cam_thread = CamThread(mutex=self.mutex, condition=self.condition)
        self.cam_thread.signal_im_ready.connect(self.set_im)
        self.cam_thread.signal_original_im_ready.connect(self.set_original_im)

        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.is_moving = False
        self.is_resizing = False

        self.is_capture = False
        self.confidence = 0.5

        """ controls """
        # layout
        self.hbox_main = QHBoxLayout(self)
        self.vbox_btns = QVBoxLayout(self)
        self.vbox_im = QHBoxLayout(self)
        self.hbox_who = QHBoxLayout(self)

        # button
        self.btn_save_image = QPushButton('Save Image', self)
        self.btn_train = QPushButton('Train', self)
        self.btn_load_image = QPushButton('Load Image \n And Recognize', self)
        self.btn_start_camera = QPushButton('Start Camera', self)
        self.btn_stop_camera = QPushButton('Stop Camera', self)
        self.btn_test = QPushButton('Test', self)

        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_save_image.setSizePolicy(size_policy)
        self.btn_train.setSizePolicy(size_policy)
        self.btn_load_image.setSizePolicy(size_policy)
        self.btn_start_camera.setSizePolicy(size_policy)
        self.btn_stop_camera.setSizePolicy(size_policy)
        self.btn_test.setSizePolicy(size_policy)

        # line text
        self.txb_who = QLineEdit("", self)

        # label
        self.lb_realtime = QLabel("", self)
        self.lb_template = QLabel("", self)
        self.lb_result = QLabel("", self)
        self.lb_who = QLabel('Who:', self)

        """ place controls to layout """
        self.hbox_who.addWidget(self.lb_who)
        self.hbox_who.addWidget(self.txb_who)

        self.vbox_im.addWidget(self.lb_realtime, alignment=Qt.AlignTop)

        self.vbox_btns.addLayout(self.hbox_who)
        self.vbox_btns.addWidget(self.btn_save_image)
        self.vbox_btns.addWidget(self.btn_train)
        self.vbox_btns.addWidget(self.btn_load_image)
        self.vbox_btns.addWidget(self.btn_start_camera)
        self.vbox_btns.addWidget(self.btn_stop_camera)
        self.vbox_btns.addWidget(self.btn_test)

        self.hbox_main.addLayout(self.vbox_im)
        self.hbox_main.addLayout(self.vbox_btns)
        self.hbox_main.addWidget(self.lb_template, alignment=Qt.AlignHCenter)
        self.hbox_main.addWidget(self.lb_result, alignment=Qt.AlignHCenter)

        self.setLayout(self.hbox_main)

        """ connect signal and slot """
        self.btn_save_image.clicked.connect(self.on_btn_save_image_clicked)
        self.btn_train.clicked.connect(self.on_btn_train_clicked)
        self.btn_load_image.clicked.connect(self.on_btn_load_image_clicked)
        self.btn_start_camera.clicked.connect(self.on_btn_start_camera_clicked)
        self.btn_stop_camera.clicked.connect(self.on_btn_stop_camera_clicked)
        self.btn_test.clicked.connect(self.on_btn_test_clicked)

        self.lb_realtime.mousePressEvent = self.on_lb_realtime_mouse_press
        self.lb_realtime.mouseMoveEvent = self.on_lb_realtime_mouse_move
        self.lb_realtime.mouseReleaseEvent = self.on_lb_realtime_mouse_release

        self.setWindowTitle('qt face recognition')
        self.show()

        self.btn_test.setVisible(False)

    @pyqtSlot('QImage')
    def set_im(self, image):
        self.mutex.lock()
        try:
            self.lb_realtime.setPixmap(QPixmap.fromImage(image))
        finally:
            self.mutex.unlock()
            self.condition.wakeAll()

    @pyqtSlot(object)
    def set_original_im(self, cv_im):
        self.im_original = cv_im

    def on_lb_realtime_mouse_move(self, event):
        try:

            is_moving = True
            self.x1 = event.x()
            self.y1 = event.y()
            self.cam_thread.updateDrawingData(self.x0,
                                              self.y0,
                                              self.x1,
                                              self.y1,
                                              is_moving)

        except Exception as e:
            print(e)

    def on_lb_realtime_mouse_press(self, event):
        try:
            self.is_moving = False
            self.x0 = event.x()
            self.y0 = event.y()

            self.cam_thread.updateDrawingData(self.x0,
                                              self.y0,
                                              self.x1,
                                              self.y1,
                                              self.is_moving)
        except Exception as e:
            print(e)

    def on_lb_realtime_mouse_release(self, event):
        pass

    def closeEvent(self, event):
        print("[INFO] closed")
        self.cam_thread.update_is_capture(False)
        event.accept()

    def on_btn_load_image_clicked(self, _):
        self.load_image_and_recognize()

    def on_btn_save_image_clicked(self, _):
        if self.is_capture:
            self.save_image()
        else:
            print("[INFO] please start camera first")

    def on_btn_train_clicked(self, _):
        self.is_capture = False
        self.cam_thread.update_is_capture(self.is_capture)

        self.train()

    def on_btn_start_camera_clicked(self, _):
        if not self.is_capture:
            self.mutex.lock()

            self.is_capture = True
            self.cam_thread.update_is_capture(self.is_capture)

            self.cam_thread.my_face_recognizer.load_face_recognize_models()
            self.cam_thread.my_face_recognizer.load_face_detect_model()

            self.cam_thread.start()

    def on_btn_stop_camera_clicked(self, _):
        self.is_capture = False
        self.cam_thread.update_is_capture(self.is_capture)

    def on_btn_test_clicked(self, _):
        # self.cam_thread.my_face_recognizer.load_face_detect_model()
        # self.cam_thread.my_face_recognizer.load_face_recognize_models()
        pass

    def load_image_and_recognize(self):
        try:
            filter_by = "Image files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"

            fname = QFileDialog.getOpenFileName(self,
                                                "Open file",
                                                os.getcwd(),
                                                filter_by)[0]

            if not fname == "":

                img = cv2.imread(fname, cv2.IMREAD_COLOR)
                img = imutils.resize(img, width=600)
                (h, w) = img.shape[:2]

                image_blob = cv2.dnn.blobFromImage(
                    cv2.resize(img, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)

                face_recognizer = FaceRecognizer()
                face_recognizer.load_face_recognize_models()
                face_recognizer.load_face_detect_model()

                detected_faces = face_recognizer.detect_faces(image_blob)
                index_face = 0
                confidence = detected_faces[0, 0, index_face, 2]

                if confidence > face_recognizer.get_min_confidence():
                    box = detected_faces[0, 0, index_face, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI
                    face = img[startY:endY, startX:endX]
                    # (fH, fW) = face.shape[:2]

                    # # ensure the face width and height are sufficiently large
                    # if fW < 20 or fH < 20:
                    #    continue

                    face_blob = cv2.dnn.blobFromImage(face,
                                                      1.0 / 255,
                                                      (96, 96),
                                                      (0, 0, 0),
                                                      swapRB=True,
                                                      crop=False)

                    name, proba = face_recognizer.predict_name(face_blob)
                    self.txb_who.setText(name)

                    # draw the bounding box and probability
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10

                    img = cv2.cvtColor(img,
                                       cv2.COLOR_BGR2RGB)

                    cv2.rectangle(img, (startX, startY), (endX, endY),
                                  (255, 0, 0), 2)
                    cv2.putText(img, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 0, 0), 2)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                q_img = QImage(img.data,
                               img.shape[1],
                               img.shape[0],
                               QImage.Format_RGB888)
                q_img = q_img.scaled(640, 480,
                                     Qt.KeepAspectRatio)

                self.lb_realtime.setPixmap(QPixmap.fromImage(q_img))
                self.lb_realtime.update()
        except Exception as e:
            print(e)

    def save_image(self):

        folder_path = self.cur_path + '/face_recognition_dataset/' + self.txb_who.text()
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        num_images = 0
        for (dirpath, dirnames, filenames) in walk(folder_path):
            for filename in filenames:
                filetype = filename.split(".")[-1]
                if filetype == "jpg":
                    num_images += 1

        str_number = '{:0>5d}'.format(num_images)
        image_path = folder_path + '/' + self.txb_who.text() + str_number + '.jpg'
        cv2.imwrite(image_path, self.im_original)
        print("[INFO] saved " + image_path)

    def train(self):
        self.cam_thread.my_face_recognizer.train_with_bo()
        # self.cam_thread.my_face_recognizer.train()


class CamThread(QThread):
    signal_im_ready = pyqtSignal('QImage')
    signal_original_im_ready = pyqtSignal(object)

    def __init__(self, mutex, condition):
        super(CamThread, self).__init__()
        self.mutex = mutex
        self.condition = condition
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.is_moving = False
        self.is_capture = False

        self.my_face_recognizer = FaceRecognizer()

    def update_is_capture(self, is_capture):
        self.is_capture = is_capture

    def updateDrawingData(self, x0, y0, x1, y1, is_moving):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.is_moving = is_moving

    def run(self):

        print("[INFO] starting cam")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ret = True
        while ret:
            try:
                if not self.is_capture:
                    cap.release()
                    break

                ret, frame = cap.read()

                if ret:
                    frame = imutils.resize(frame, width=600)
                    (h, w) = frame.shape[:2]

                    self.signal_original_im_ready.emit(frame.copy())

                    # draw roi
                    if self.is_moving:
                        frame = cv2.rectangle(frame,
                                              pt1=(self.x0, self.y0),
                                              pt2=(self.x1, self.y1),
                                              color=(0, 255, 0),
                                              thickness=2)

                    image_blob = cv2.dnn.blobFromImage(
                        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                        (104.0, 177.0, 123.0), swapRB=False, crop=False)

                    # detector face
                    detected_faces = self.my_face_recognizer.detect_faces(image_blob)

                    index_face = 0
                    confidence = detected_faces[0, 0, index_face, 2]
                    # filter out weak detections
                    if confidence > self.my_face_recognizer.get_min_confidence():
                        box = detected_faces[0, 0, index_face, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # extract the face ROI
                        face = frame[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]

                        # ensure the face width and height are sufficiently large
                        if fW < 20 or fH < 20:
                            continue

                        face_blob = cv2.dnn.blobFromImage(face,
                                                          1.0 / 255,
                                                          (96, 96),
                                                          (0, 0, 0),
                                                          swapRB=True,
                                                          crop=False)

                        name, proba = self.my_face_recognizer.predict_name(face_blob)

                        # draw the bounding box and probability
                        text = "{}: {:.2f}%".format(name, proba * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10

                        frame = cv2.cvtColor(frame,
                                             cv2.COLOR_BGR2RGB)

                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      (255, 0, 0), 2)
                        cv2.putText(frame, text, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (255, 0, 0), 2)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # show image
                    q_img = QImage(frame.data,
                                   frame.shape[1],
                                   frame.shape[0],
                                   QImage.Format_RGB888)
                    q_img = q_img.scaled(640, 480,
                                         Qt.KeepAspectRatio)

                    self.signal_im_ready.emit(q_img)

                    self.condition.wait(self.mutex)

            except Exception as e:
                print(e)

        self.mutex.unlock()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWidget()
    sys.exit(app.exec_())
