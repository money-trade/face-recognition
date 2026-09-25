import cv2
import os
import numpy as np

def knn(X, y, z, k=3):
    d = np.sum((X - z) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)]

# โหลดรูปจากแต่ละโฟลเดอร์ (ชื่อโฟลเดอร์ = ชื่อคน)
X_train, y_train = [], []
for f in os.listdir():
    if os.path.isdir(f):
        for i in os.listdir(f):
            if i.endswith(".jpg"):
                x = cv2.imread(f"{f}/{i}", cv2.IMREAD_GRAYSCALE)
                if x is not None:
                    X_train.append(x.flatten())
                    y_train.append(f)

X_train = np.array(X_train)
y_train = np.array(y_train)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print('เปิดกล้องไม่ได้')
        break

    cv2.rectangle(frame, (250, 120), (390, 300), (0, 0, 255), 2)
    face = cv2.cvtColor(frame[120:300, 250:390], cv2.COLOR_BGR2GRAY)

    if len(X_train) > 0:
        name = knn(X_train, y_train, face.flatten(), k=3)
        cv2.putText(frame, name, (250, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()