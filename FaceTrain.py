import cv2
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
name = input('ชื่อคนที่จะถ่าย: ').strip()
os.makedirs(name, exist_ok=True)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
i = len(os.listdir(name)) + 1
try:
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
        if key == ord('s'):
            cv2.imwrite(f'{name}/{i}.jpg', face)
            print(f'บันทึก {name}/{i}.jpg')
            i += 1
        # q / Esc / กด X ปิดหน้าต่าง
        elif key in (ord('q'), 27) or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()