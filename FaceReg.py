import cv2
import os
import numpy as np

def knn(X, y, z, k=3):
    d = np.sum((X - z) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)]

os.chdir(os.path.dirname(os.path.abspath(__file__)))
X, y = [], []
for f in os.listdir():
    if os.path.isdir(f) and not f.startswith('.'):
        for i in os.listdir(f):
            if i.endswith('.jpg'):
                X.append(cv2.imread(f'{f}/{i}', cv2.IMREAD_GRAYSCALE).flatten())
                y.append(f)
X = np.array(X, dtype=float)
y = np.array(y)

# CAP_DSHOW เปิดกล้องเร็วกว่าบน Windows ส่วนระบบอื่นใช้ค่าเริ่มต้น
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print('เปิดกล้องไม่ได้')
            break
        cv2.rectangle(frame, (250,120), (390,300), (0,0,255), 2)
        face = cv2.cvtColor(frame[120:300, 250:390, :], cv2.COLOR_BGR2GRAY)
        if len(X) > 0:
            name = knn(X, y, face.flatten())
            cv2.putText(frame, name, (250,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        # q / Esc / กด X ปิดหน้าต่าง
        if key in (ord('q'), 27) or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
