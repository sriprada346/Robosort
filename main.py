from  fastapi import FastAPI,Query
import uvicorn
import cv2
import mediapipe as mp
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import pathlib
temp = pathlib.WindowsPath
pathlib.PosixPath = pathlib.WindowsPath
import torch
import threading

app = FastAPI()
model = torch.hub.load('ultralytics/yolov5','custom',path='best.pt')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

current_mode = None
mode_thread = None
mode_stop_event = threading.Event()

# x_min,x_mid,x_max = 0,75,180
# palm_angle_min,palm_angle_mid = -50,20

# y_min,y_mid,y_max = 0,90,180
# wrist_y_min,wrist_y_max = 0.3,0.9

# z_min,z_mid,z_max = 10,90,180
# palm_size_min,palm_size_max = 0.1,0.3
    
# claw_open_angle,claw_close_angle = 50,0

###########################################

x_min,x_mid,x_max = 0,75,180
palm_angle_min,palm_angle_mid = -50,20

y_min,y_mid,y_max = 0,90,180
wrist_y_min,wrist_y_max = 0.3,0.9

z_min,z_mid,z_max = 10,90,180
palm_size_min,palm_size_max = 0.1,0.3
    
claw_open_angle,claw_close_angle = 50,0

servo_angle = [x_mid,y_mid,z_mid,claw_open_angle]
prev_servo_angle = servo_angle
fist_threshold = 7

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
map_range = lambda x, in_min, in_max, out_min, out_max: abs((x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min)

def is_fist(hand_landmarks,palm_size):
    # we are going to calculate distance between wrist and each finger tips 
    distance_sum = 0
    WRIST = hand_landmarks.landmark[0]
    for i in [7,8,10,11,15,16,19,20]:
        distance_sum += ((WRIST.x - hand_landmarks.landmark[i].x)**2 + \
                        (WRIST.y - hand_landmarks.landmark[i].y)**2 + \
                        (WRIST.z - hand_landmarks.landmark[i].z)**2)**0.5
    return distance_sum/palm_size < fist_threshold

def landmark_to_servo_angle(hand_landmarks):
    global servo_angle 
    WRIST = hand_landmarks.landmark[0]
    INDEX_FINGER_MCP = hand_landmarks.landmark[5]
    palm_size = ((WRIST.x - INDEX_FINGER_MCP.x)**2 + (WRIST.y - INDEX_FINGER_MCP.y) ** 2 + (WRIST.z - INDEX_FINGER_MCP.z)**2)**0.5

    if is_fist(hand_landmarks,palm_size):
        servo_angle[3] = claw_close_angle
    else:
        servo_angle[3] = claw_open_angle

    distance = palm_size

    #x-angle
    angle = (WRIST.x - INDEX_FINGER_MCP.x) / distance
    angle = int(angle * 180 / 3.1415926)
    angle = clamp(angle,palm_angle_min,palm_angle_mid)
    servo_angle[0] = map_range(angle,palm_angle_min,palm_angle_mid,x_max,x_min)

    #y-angle
    wrist_y = clamp(WRIST.y,wrist_y_min,wrist_y_max)
    servo_angle[1] = map_range(wrist_y,wrist_y_min,wrist_y_max,y_max,y_min)

    #z-angle
    palm_size = clamp(palm_size,palm_size_min,palm_size_max)
    servo_angle[2] = map_range(palm_size,palm_size_min,palm_size_max,z_max,z_min)

    servo_angle = [int(i) for i in servo_angle]

    return servo_angle

def run_mediapipe():
   
    mode_stop_event.clear()
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    while not mode_stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print('Failed to capture frame')
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                servo_angle = landmark_to_servo_angle(hand_landmarks)
                print(servo_angle)
        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(1) == 27:  # ESC key
            break
    cap.release()
    cv2.destroyAllWindows()

def run_yolo():
    print(f"Starting YOLO Thread {threading.current_thread().name}")
    mode_stop_event.clear()
    cap = cv2.VideoCapture(1)
    while not mode_stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print('Failed to capture frame')
            break
        frame = cv2.flip(frame, 1)
        results = model(frame)
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('YOLO', frame)
        if cv2.waitKey(1) == 27:  # ESC key
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"YOLO thread {threading.current_thread().name} stopped")

def stop_current_mode():
    global mode_thread,current_mode

    if mode_thread and mode_thread.is_alive():
        print(f"Stopping {current_mode} mode thread: {mode_thread.name}")
        mode_stop_event.set()
        mode_thread.join()

@app.get('/control')
async def myapi(mode : str=Query(...)):
    global current_mode,mode_thread
    if mode == current_mode:
        return {"message": f"Already in {mode} mode"}
    
    stop_current_mode()
    current_mode = mode

    if mode == 'manual':
        print("Switched to Manual Mode")
    elif mode == 'semi':
        print("Switched to Semi Mode , Models are Loading")
        mode_thread = threading.Thread(target=run_mediapipe,name='MediaPipeThread')
        mode_thread.start()
        print("Thread is created ")
    elif mode == 'auto':
        mode_thread = threading.Thread(target=run_yolo,name='YOLOThread')
        mode_thread.start()
        print(mode)
    elif mode == 'stop':
        stop_current_mode()
        current_mode = None
        print(mode)
    return {"message": f"Switched to {mode} mode"}

@app.get('/slider')
async def myapi(servo1 : int= Query(...),servo2:int=Query(...),servo3 : int=Query(...),servo4:int=Query(...)):
    servo1+=90
    print(servo1)
    print(servo2)
    print(servo3)
    print(servo4)


    return {
        "servo1": servo1,
        "servo2": servo2,
        "servo3": servo3,
        "servo4": servo4
    }


if __name__ == '__main__':
    uvicorn.run(app,host='0.0.0.0',port=8000)
