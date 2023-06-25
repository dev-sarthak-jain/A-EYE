import cv2
from facial_emotion_recognition import EmotionRecognition
import mediapipe as mp
import os
import speech_recognition as sr
from gtts import gTTS

classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)

    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(
                        img,
                        classNames[classId - 1].upper(),
                        (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        img,
                        str(round(confidence * 100, 2)),
                        (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

    return img, objectInfo


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    er = EmotionRecognition(device="cpu")
    while True:
        success, frame = cap.read()
        result, objectInfo = getObjects(frame, 0.45, 0.2)
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        frame = er.recognise_emotion(frame, return_type="BGR")
        print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for handLandMarks in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLandMarks, mpHands.HAND_CONNECTIONS)
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
