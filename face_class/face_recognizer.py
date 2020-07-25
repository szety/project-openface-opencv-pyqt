"""
A simple face recognition class that detects faces, trains and
recoginizes faces.
"""

# Author: Tom Sze <sze.takyu@gmail.com>

import os
import pickle
import cv2
import imutils
import numpy as np
from imutils import paths
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from termcolor import colored
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def optimize_xgb(data, targets, n_iter, cv):
    """Apply Bayesian Optimization to xgb.

    This optimizes for hyperparameters of xgb.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The data to fit.

    targets : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    n_iter : int
        The number of iterations of optimization.

    Returns
    -------
    best_param : dict
        Result of best hyperparameters after optimization.
    """

    def xgb_crossval(max_depth, learning_rate):
        """Wrapper of xgb cross validation.
        """
        return xgb_cv(
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            data=data,
            targets=targets,
            cv=cv
        )

    optimizer = BayesianOptimization(
        f=xgb_crossval,
        pbounds={
            "max_depth": (0, 5),
            "learning_rate": (0.1, 1)
        },
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=n_iter)

    print("Best parameters:", optimizer.max['params'])
    return optimizer.max['params']


def xgb_cv(max_depth, learning_rate, data, targets, cv):
    """Evaluate cross-validation score of AUC of a xgb classifier

    A pipeline standardizes data and fits data to a xgb classifier with
    given max_depth and learning_rate parameters. The cross validation
    score is found on this pipeline with given data and targets.

    Parameters
    ----------
    max_depth :

    learning_rate :

    data : array-like of shape (n_samples, n_features)
        The data to fit.

    targets : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    Returns
    -------
    score : numpy.float64
        Mean AUC score of the estimator for each run of the cross validation.
    """

    # Create the classifier with new parameters in a pipeline
    pipe = make_pipeline(StandardScaler(),
                         XGBClassifier(learning_rate=learning_rate,
                                       max_depth=max_depth))
    cval = cross_val_score(pipe,
                           data,
                           targets,
                           cv=5,
                           scoring='accuracy')

    return cval.mean()


class FaceRecognizer:

    def __init__(self):

        self._min_confidence = 0
        self._current_path = os.getcwd()
        self._face_recognition_dataset_path = os.path.join(self._current_path,
                                                           "face_recognition_dataset")
        self._pretrained_model_path = os.path.join(self._current_path,
                                                   "pretrained_models")

        self._proto_path = os.path.join(self._pretrained_model_path,
                                        "deploy.prototxt")
        self._face_det_model_path = os.path.join(self._pretrained_model_path,
                                                 "res10_300x300_ssd_iter_140000.caffemodel")
        self._face_embedder_path = os.path.join(self._pretrained_model_path,
                                                "openface_nn4.small2.v1.t7")

        self._custom_trained_model_path = os.path.join(self._current_path,
                                                       "custom_trained_model")
        self._face_recognizer_path = os.path.join(self._custom_trained_model_path,
                                                  "recognizer.pickle")
        self._label_encoder_path = os.path.join(self._custom_trained_model_path,
                                                "le.pickle")
        self._face_embedding_path = os.path.join(self._custom_trained_model_path,
                                                 "embeddings.pickle")

        self._face_detector = None
        self._face_embedder = None
        self._face_recognizer = None
        self._label_encoder = None

        self._create_folders()

    def _create_folders(self):
        """Create folders at the calling script location"""
        os.makedirs(self._custom_trained_model_path, exist_ok=True)
        os.makedirs(self._face_recognition_dataset_path, exist_ok=True)

    def get_min_confidence(self):
        """"Get the confidence after detecting faces"""
        return self._min_confidence

    def detect_faces(self, image_blob):
        """Detect faces in an image blob

        Returns array contains confidence of each face and bounding box.

        Parameters
        ----------
        image_blob : numpy ndarray of float, shape=(1,channel,width,height)
            The image for detecting faces fron :function cv2.dnn.blobFromImage.

        Returns
        -------
        detected_faces : numpy ndarray of float, shape=(1,1,200,7)
            Result of detected faces:
                for nth face's confidence : detected_faces[1,1,n,2].
                for nth face's bounding box : detected_faces[1,1,n,3:7] which is in ratio.
        """
        self._face_detector.setInput(image_blob)
        detected_faces = self._face_detector.forward()

        return detected_faces

    def predict_name(self, face_blob):
        """Predict the person's name on a face_blob

        Parameters
        ----------
        face_blob : numpy ndarray of float, shape=(1,channel,width,height)
            The image of a face from :function cv2.dnn.blobFromImage.

        Returns
        -------
        name : str
            The Name of the recognized face.
        proba : float
            The probability of the recognized face.
        """

        # Feed the face to face embedder for 128-d embbeddings
        self._face_embedder.setInput(face_blob)
        vec = self._face_embedder.forward()

        # Classify the 128-d embedding
        preds = self._face_recognizer.predict_proba(vec)[0]

        # Get the name of highest prediction probability
        j = np.argmax(preds)
        proba = preds[j]
        name = self._label_encoder.classes_[j]

        return name, proba

    def load_face_detect_model(self,
                               proto_path=None,
                               face_det_model_path=None):

        """Load a caffe face detector model from given path

        Parameters
        ----------
        proto_path : str
            The path to the proto file of the face detector

        face_det_model_path : str
            The path to the face detector model file.

        Returns
        -------
        None
        """
        try:
            print("[INFO] loading face detector")
            if proto_path is None and face_det_model_path is None:
                proto_path = self._proto_path
                face_det_model_path = self._face_det_model_path

            self._face_detector = cv2.dnn.readNetFromCaffe(proto_path,
                                                           face_det_model_path)
            print("[INFO] done loading face detector")
        except Exception as e:
            print(e)

    def load_face_recognize_models(self,
                                   face_embedder_path=None,
                                   face_recognizer_path=None,
                                   label_encoder_path=None):

        """Load face recognize embedder, model, label encoder

        Parameters
        ----------
        face_embedder_path : str
            The path to the face embedder model.
        face_recognizer_path : str
            The path to the face recognizer model trainined.
        label_encoder_path : str
            The path to the label enconder corresponding to the dataset.
        Returns
        -------
        None
        """
        try:
            print("[INFO] loading face embedder, recognizer, label encoder")
            if face_recognizer_path is None or \
                    face_recognizer_path is None or \
                    label_encoder_path is None:
                face_embedder_path = self._face_embedder_path
                face_recognizer_path = self._face_recognizer_path
                label_encoder_path = self._label_encoder_path

            self._face_embedder = cv2.dnn.readNetFromTorch(face_embedder_path)
            self._face_recognizer = pickle.loads(open(face_recognizer_path, "rb").read())
            self._label_encoder = pickle.loads(open(label_encoder_path, "rb").read())

            print("[INFO] done loading face embedder, recognizer, label encoder")
        except Exception as e:
            print(e)

    def train(self,
              min_confidence=None,
              face_recognition_dataset_path=None,
              face_embedding_path=None,
              face_recognizer_path=None,
              label_encoder_path=None):

        """Train the face recognizer

        This funciton loads the face detector model and the face embbeder model.
        The face embedder model extracts the 128-d face embeddings from each face image.
        A xgboost classifier fits those 128-d face embeddings and classify them.
        The resultant classifier is saved.

        Parameters
        ----------
        min_confidence : float
            The minimun confidence to detect a face. Face detected with confidence
            higher than this will be used to extract 128-d embeddings.
        face_recognition_dataset_path : str
            The path to face dataset. Structure is like:
            dataset_folder  - personA_folder - images
                            - personB_folder - images
        face_embedding_path : str
            The path to save the face embeddings
        face_recognizer_path : str
            The path to save the face recognizer
        label_encoder_path : str
            The path to save the label encoder
        """
        if min_confidence is None:
            min_confidence = self._min_confidence

        self.load_face_detect_model()
        self.load_face_recognize_models()

        self._extract_embedding(min_confidence,
                                face_recognition_dataset_path)
        self._train_embedding(face_embedding_path,
                              face_recognizer_path,
                              label_encoder_path)

    def _extract_embedding(self,
                           min_confidence=0.5,
                           face_recognition_dataset_path=None):
        """Extract 128-d face embedding using face embedder on face image

        This functions first detects a face on an image using the face detector.
        The face embedder model then extracts the 128-d face embeddings from each
        face image. The face embeddings and correponding person are saved as
        embeddings.pickle.

        Parameters
        ----------
        min_confidence : float
            The minimun confidence to detect a face. Face detected with confidence
            higher than this will be used to extract 128-d embeddings.
        face_recognition_dataset_path : str
                    The path to face dataset. Structure is like:
                    dataset_folder  - personA_folder - images
                                    - personB_folder - images
        Returns
        -------
        None
        """
        self._min_confidence = min_confidence
        if face_recognition_dataset_path is None:
            face_recognition_dataset_path = self._face_recognition_dataset_path
        image_paths = list(paths.list_images(face_recognition_dataset_path))

        known_embeddings = []
        known_names = []
        total = 0

        for (i, image_path) in enumerate(image_paths):
            print("[INFO] processing image {}/{} {}".format(i + 1,
                                                            len(image_paths),
                                                            image_path))
            name = image_path.split(os.path.sep)[-2]

            image = cv2.imread(image_path)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]

            image_blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            self._face_detector.setInput(image_blob)
            detections = self._face_detector.forward()

            if len(detections) > 0:
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                if confidence > self._min_confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    if fW < 20 or fH < 20:
                        continue

                    face_blob = cv2.dnn.blobFromImage(face,
                                                      scalefactor=1.0 / 255,
                                                      size=(96, 96),
                                                      mean=(0, 0, 0),
                                                      swapRB=True,
                                                      crop=False)
                    self._face_embedder.setInput(face_blob)
                    vec = self._face_embedder.forward()

                    known_names.append(name)
                    known_embeddings.append(vec.flatten())
                    total += 1

        data = {"embeddings": known_embeddings, "names": known_names}

        f = open(self._face_embedding_path, "wb")
        f.write(pickle.dumps(data))
        f.close()
        print("[INFO] done saving embeddings")

    def _train_embedding(self,
                         face_embedding_path=None,
                         face_recognizer_path=None,
                         label_encoder_path=None):
        """Train the saved embeddings using xgboost classifier

        This functions fits names of persons within saved embedding to a label encoder.
        A xgboost classifer then fits the 128-d embeddings to classify the faces.

        Parameters
        ----------
        face_embedding_path : str
            The path to save the face embeddings
        face_recognizer_path : str
            The path to save the face recognizer
        label_encoder_path : str
            The path to save the label encoder
        """
        if face_embedding_path is None:
            face_embedding_path = self._face_embedding_path
        if face_recognizer_path is None:
            face_recognizer_path = self._face_recognizer_path
        if label_encoder_path is None:
            label_encoder_path = self._label_encoder_path

        data = pickle.loads(open(face_embedding_path, "rb").read())

        print("[INFO] loading encoding labels...")
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])

        print("[INFO] training model...")
        arrays = data["embeddings"]
        data = np.stack(arrays, axis=0)
        y = labels

        face_recognizer = XGBClassifier()
        face_recognizer.fit(data, y)

        f = open(face_recognizer_path, "wb")
        f.write(pickle.dumps(face_recognizer))
        f.close()

        print("[INFO] done saving recognizer")
        f = open(label_encoder_path, "wb")
        f.write(pickle.dumps(le))
        f.close()
        print("[INFO] done saving label encoder")

    def train_with_bo(self,
                      face_embedding_path=None,
                      face_recognizer_path=None,
                      label_encoder_path=None,
                      n_iter=50):
        """Train the saved embeddings using xgboost classifier

        This functions fits names of persons within saved embedding to a label encoder.
        A xgboost classifer then fits the 128-d embeddings to classify the faces.

        Parameters
        ----------
        face_embedding_path : str
            The path to save the face embeddings
        face_recognizer_path : str
            The path to save the face recognizer
        label_encoder_path : str
            The path to save the label encoder
        """
        if face_embedding_path is None:
            face_embedding_path = self._face_embedding_path
        if face_recognizer_path is None:
            face_recognizer_path = self._face_recognizer_path
        if label_encoder_path is None:
            label_encoder_path = self._label_encoder_path

        data = pickle.loads(open(face_embedding_path, "rb").read())
        print("len(data.names):" + str(len(data["names"])))

        print("[INFO] loading encoding labels...")
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])

        print("[INFO] training model...")
        arrays = data["embeddings"]
        data = np.stack(arrays, axis=0)
        y = labels

        # =============

        seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(data, y,
                                                            test_size=test_size,
                                                            random_state=seed)

        print(colored('--- Optimizing xgb ---', 'green'))
        best_params = optimize_xgb(X_train, y_train, n_iter=n_iter, cv=8)

        face_recognizer = make_pipeline(StandardScaler(),
                                        XGBClassifier(learning_rate=best_params['learning_rate'],
                                                      max_depth=int(best_params['max_depth'])))

        face_recognizer.fit(data, y)
        # =============

        f = open(face_recognizer_path, "wb")
        f.write(pickle.dumps(face_recognizer))
        f.close()

        print("[INFO] done saving recognizer")
        f = open(label_encoder_path, "wb")
        f.write(pickle.dumps(le))
        f.close()
        print("[INFO] done saving label encoder")
