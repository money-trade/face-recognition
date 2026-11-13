import cv2
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K = 1                  # จำนวนรูปที่ใกล้ที่สุดที่ใช้โหวต (เลือกจาก evaluate.py: k=1 แม่นที่สุด)
FACE_SHAPE = (180, 140)  # (สูง, กว้าง) ของรูปที่ FaceTrain.py บันทึก


def knn(X, y, z, k=K):
    d = np.sum((X - z) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)]


def rms(d2):
    """ระยะกำลังสองรวม -> ค่าพิกเซลต่างกันเฉลี่ย (0-255) อ่านง่ายกว่า"""
    return np.sqrt(np.maximum(d2, 0) / (FACE_SHAPE[0] * FACE_SHAPE[1]))


def two_closest(X, y, z):
    """คนที่ใกล้ที่สุด กับคนที่ใกล้เป็นอันดับ 2 (คนละคนกัน) พร้อมระยะ
    ถ้าสองระยะใกล้กันมาก แปลว่าระบบไม่แน่ใจ"""
    d = rms(np.sum((X - z) ** 2, axis=1))
    best = {}
    for name in np.unique(y):
        best[name] = d[y == name].min()
    ranked = sorted(best.items(), key=lambda t: t[1])
    return ranked[0], (ranked[1] if len(ranked) > 1 else None)


def load_faces(folder=HERE):
    """อ่านรูปจากทุกโฟลเดอร์ (ชื่อโฟลเดอร์ = ชื่อคน) ข้ามไฟล์ที่อ่านไม่ได้หรือขนาดไม่ตรง"""
    X, y = [], []
    for f in sorted(os.listdir(folder)):
        path = os.path.join(folder, f)
        if not os.path.isdir(path) or f.startswith(('.', '_')):
            continue
        for i in sorted(os.listdir(path)):
            if not i.lower().endswith('.jpg'):
                continue
            img = cv2.imread(os.path.join(path, i), cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape != FACE_SHAPE:
                print(f'  ข้าม {f}/{i}: อ่านไม่ได้หรือขนาดไม่ใช่ {FACE_SHAPE[1]}x{FACE_SHAPE[0]}')
                continue
            X.append(img.flatten())
            y.append(f)
    return np.array(X, dtype=float), np.array(y)


def main():
    X, y = load_faces()
    if len(X) == 0:
        print('ยังไม่มีรูปใบหน้า ให้รัน FaceTrain.py ก่อน')
        return
    names, counts = np.unique(y, return_counts=True)
    print(f'โหลด {len(X)} รูป จาก {len(names)} คน: ' + ', '.join(f'{n} {c}' for n, c in zip(names, counts)))

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
            z = face.flatten().astype(float)
            name = knn(X, y, z)
            cv2.putText(frame, name, (250,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # ใต้กรอบ: ระยะถึงคนที่ใกล้สุด และคนที่ใกล้อันดับ 2 (ยิ่งห่างกันมาก ยิ่งมั่นใจ)
            (n1, d1), second = two_closest(X, y, z)
            info = f'd={d1:.1f}'
            if second:
                info += f'  2nd: {second[0]} d={second[1]:.1f}'
                if second[1] - d1 < 2:   # ต่างกันไม่ถึง 2 = ก้ำกึ่ง
                    info += '  (unsure)'
            cv2.putText(frame, info, (180,325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1) & 0xFF
            # q / Esc / กด X ปิดหน้าต่าง
            if key in (ord('q'), 27) or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
