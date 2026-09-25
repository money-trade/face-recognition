import cv2
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
name = 'chirapchinyo'
os.makedirs(name, exist_ok=True)
i = len(os.listdir(name)) + 1
while True:
    ret, frame = cap.read()
    if not ret:
        print('เปิดกล้องไม่ได้')
        break
    cv2.rectangle(frame, (250,120), (390,300), (0,0,255), 2)
    face = cv2.cvtColor(frame[120:300, 250:390, :], cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('face', face)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('s'), ord('S')):
        cv2.imwrite(f'{name}/{i}.jpg', face)
        print(f'บันทึกรูป {name}/{i}.jpg')
        i += 1
    elif key in (ord('q'), ord('Q')):
        break
cap.release()
cv2.destroyAllWindows()