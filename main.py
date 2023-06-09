import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
from tensorflow.keras.models import load_model

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# To access the speaker through the pycaw library
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volbar = 400
volper = 0

volMin, volMax = volume.GetVolumeRange()[:2]

while True:
    success, img = cap.read()  # Capture an image from the camera
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Hand detection
    results = hands.process(imgRGB)  # Process the image

    lmList = []  # Empty list
    if results.multi_hand_landmarks:  # Check if hands are detected
        for handLandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handLandmark, mpHands.HAND_CONNECTIONS)

    if lmList:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger

        cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)  # Draw circles on thumb and index finger tips
        cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Draw a line between thumb and index finger tips

        length = hypot(x2 - x1, y2 - y1)  # Distance between thumb and index finger tips
        threshold = 100  # Set your own threshold value
        input_size = (21, 2)

        # Gesture recognition
        # Gesture recognition
        if length > threshold:
            # Perform gesture recognition
            cropped_img = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            resized_img = cv2.resize(cropped_img, (input_size[1], input_size[0]))
            normalized_img = resized_img / 255.0
            reshaped_img = np.expand_dims(normalized_img, axis=0)
            reshaped_img = np.transpose(reshaped_img, (0, 2, 1, 3))
            prediction = model.predict(reshaped_img)

            classID = np.argmax(prediction)
            className = classNames[classID]

            # Show the predicted gesture on the frame
            cv2.putText(img, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


        # Convert the length to volume level
        vol = np.interp(length, [30, 350], [volMin, volMax])
        volbar = np.interp(length, [30, 350], [400, 150])
        volper = np.interp(length, [30, 350], [0, 100])

        print(vol, int(length))
        volume.SetMasterVolumeLevel(vol, None)

        # Creating volume bar for volume level
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
        cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)

    cv2.imshow('Image', img)  # Show the video
    if cv2.waitKey(1) & 0xFF == ord(' '):  # Press spacebar to exit
        break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close the window
