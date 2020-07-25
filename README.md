# face-recognition-openface-opencv-pyqt
This is a mini project on face recognition using OpenFace neural network model, opencv and pyqt for the user interface. Thanks Adrian Rosebrock for the tutorial on https://www.pyimagesearch.com/.



### How it works

A human face is detected using pretrained caffe face detector which implements single-shot detector (SSD) with Resnet (res10_300x300_ssd_iter_140000.caffemodel) . The face image is then feed to a pretrained OpenFace neural network model (openface_nn4.small2.v1.t7) which generates a 128-d face embedding.

To train a model, I used xgboost classifier with Bayesian optimization on those generated 128-d face embeddings from the custom dataset.

To recognize a face. The trained xgboost classifier classifies a newly generated face embedding to a person name.

A general face cognition workflow is shown as follow:

![Alt text](/images/workflow.png "Optional title")



### Install

Make sure you are using python 3.6.10. and make a webcam.

```
pip install -r requirements.txt
```



### Run

```
python qt_face_recognition.py
```

Press "Start Camera" to start recognizing faces.

![Alt text](/images/GUI.png "Optional title")

To train, type in the person's name in the text edit field next to "Who" and press "Save Image" while the camera is started. It is better to save at least 5-10 images. Then press "Train" to start training the model.



