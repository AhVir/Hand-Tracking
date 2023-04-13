import cv2 as cv
import mediapipe as mp
import time

vd = cv.VideoCapture(0)

mpHands =mp.solutions.hands
hands = mpHands.Hands()  #only uses RGB images
mpDraw = mp.solutions.drawing_utils


while True:
    isTrue, frame = vd.read()

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #conversion from bgr2rgb
    result = hands.process(rgb)
    print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handLandmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLandmarks,mpHands.HAND_CONNECTIONS)      #we are showing the BGR frames, not the RGB one.That's why, passing BGR image


    cv.imshow('Cam', frame)
    cv.waitKey(1)