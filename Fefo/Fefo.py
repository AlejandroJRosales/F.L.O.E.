import datetime
from imutils.video import FPS
import face_recognition
import cv2
import time
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input


class Emotions:
    def __init__(self):
        # parameters for loading data and images
        self.emotion_model_path = './models/emotion_model.hdf5'
        self.emotion_labels = get_labels('fer2013')

        # hyper-parameters for bounding boxes shape
        self.frame_window = 10
        self.emotion_offsets = (20, 40)

        # loading models
        self.face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
        self.emotion_classifier = load_model(self.emotion_model_path)

        # getting input model shapes for inference
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]

        # starting lists for calculating modes
        self.emotion_window = []

    def emotions(self, x, y):
        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, self.emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, self.emotion_target_size)
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = self.emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = self.emotion_labels[emotion_label_arg]
            self.emotion_window.append(emotion_text)

            if len(self.emotion_window) > self.frame_window:
                self.emotion_window.pop(0)
            try:
                emotion_mode = mode(self.emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((0, 255, 255))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((255, 255, 0))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            cv2.putText(frame, emotion_mode, (x, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def get_face_encodings():
    print("[INFO] Fetching images...")
    known_face_names = [
        "Alejandro",
        "Chelsey",
    ]

    # create arrays of known face encodings and their names
    known_face_encodings = []

    # load a sample picture and learn how to recognize it.
    print("[INFO] Encoding images...")
    for name in known_face_names:
        # Load a second sample picture and learn how to recognize it.
        folder = "faces_db/"
        print("\tEncoding " + name + ".jpg")
        known_face_encodings.append(
            face_recognition.face_encodings(face_recognition.load_image_file(folder + name + ".jpg")))
        print("\t" + name + " encoded")

    return known_face_names, known_face_encodings


def object_detection():
    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box face_bounding_box_colors for each class
    obj_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    obj_bounding_box_colors = np.random.uniform(0, 255, size=(len(obj_classes), 3))

    # load our serialized model from disk
    print("[INFO] Loading model...")
    prototxt = "real_time_object_detection_db/MobileNetSSD_deploy.prototxt.txt"
    model = "real_time_object_detection_db/MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    return obj_classes, obj_bounding_box_colors, net


def record(video_capture):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    name = datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    out = cv2.VideoWriter("recordings_db/" + name + '.mkv', fourcc, 4, (int(video_capture.get(3)), int(video_capture.get(4))))

    return out

# record video capture if true
RECORD = False

# initialize some variables
emotions = Emotions()
obj_classes, obj_bounding_box_colors, net = object_detection()
known_face_names, known_face_encodings = get_face_encodings()
face_locations = []
face_encodings = []
face_names = []
face_bounding_box_colors = []
process_this_frame = True
s = time.time()
fps = FPS().start()


# get a reference to webcam #0 (the default one)
print("[INFO] Starting video stream...")
video_capture = cv2.VideoCapture(0)
record = record(video_capture) if RECORD else None
while True:
    # grab a single frame of video
    ret, frame = video_capture.read()
    
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Video",
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    # resize frame of video to 1/4 size for faster face recognition processing
    special_number = 5
    small_frame = cv2.resize(frame, (0, 0), fx=1/special_number, fy=1/special_number)

    # convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # convert the image from BGR color to gray for emotion recognition
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # facial landmarks print
    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

    # only process every other frame of video to save time
    match = [False]
    if process_this_frame:
        # find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_bounding_box_colors = []
        for face_encoding in face_encodings:
            # see if the face is a match for the known face(s)
            name = "Unknown"
            color = (0, 0, 255)
            for i in range(len(known_face_encodings)):
                match = face_recognition.compare_faces(known_face_encodings[i], face_encoding)
                if True in match:
                    name = known_face_names[i]
                    color = (0, 128, 0)

            face_bounding_box_colors.append(color)
            face_names.append(name)

    process_this_frame = not process_this_frame

    # display the results
    index = 0
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= special_number
        right *= special_number
        bottom *= special_number
        left *= special_number

        # draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), face_bounding_box_colors[index], 1)

        # draw a label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (right + 6, top - 12), font, 1.0, face_bounding_box_colors[index], 2)

        emotions.emotions(left, top) if len(face_names) < 2 else None

        index += 1

    index2 = 0
    for face_landmarks in face_landmarks_list:

        # print the location of each facial feature in this image
        facial_features = [
            'chin',
            'left_eyebrow',
            'right_eyebrow',
            'nose_bridge',
            'nose_tip',
            'left_eye',
            'right_eye',
            'top_lip',
            'bottom_lip'
        ]

        try:
            for facial_feature in facial_features:
                for i in range(len(face_landmarks[facial_feature]) - 1):
                    lineThickness = 2
                    top = face_landmarks[facial_feature][i][0] * special_number, face_landmarks[facial_feature][i][1] * special_number
                    bottom = face_landmarks[facial_feature][i + 1][0] * special_number, face_landmarks[facial_feature][i + 1][1] * special_number
                    cv2.line(frame, top, bottom, face_bounding_box_colors[index2], lineThickness)
            index2 += 1
        except IndexError:
            pass

    # bbject detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > .25:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(obj_classes[idx],
                                         confidence * 100)

            if obj_classes[idx] == "person":
                color = (128, 0, 128)

            else:
                color = obj_bounding_box_colors[idx]

            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15

            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # display the resulting image
    cv2.imshow('Video', frame)

    if RECORD:
        record.write(frame)

    fps.update()

    # hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fps.stop()
print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

# out.release()
cv2.destroyAllWindows()
