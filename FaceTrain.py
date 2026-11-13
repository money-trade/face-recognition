import cv2
import os
import re

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ถามชื่อจนกว่าจะได้ชื่อที่ใช้ได้ (เดิมกด Enter เปล่า ๆ แล้ว error)
while True:
    name = input('ชื่อคนที่จะถ่าย: ').strip()
    if not name:
        print('  กรุณาพิมพ์ชื่อก่อนกด Enter')
    elif not re.fullmatch(r'[A-Za-z0-9_\-]+', name):
        print('  ใช้ได้เฉพาะตัวอังกฤษ ตัวเลข _ และ - (OpenCV แสดงภาษาไทยบนกล้องไม่ได้)')
    else:
        break
os.makedirs(name, exist_ok=True)

# CAP_DSHOW เปิดกล้องเร็วกว่าบน Windows ส่วนระบบอื่นใช้ค่าเริ่มต้น
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)

# เลขรูปถัดไป = เลขที่มากที่สุดในโฟลเดอร์ + 1
# (เดิมใช้จำนวนไฟล์ + 1 ถ้าเคยลบรูปกลาง ๆ ออก จะเขียนทับรูปเลขท้ายสุด)
nums = [int(f[:-4]) for f in os.listdir(name) if f.endswith('.jpg') and f[:-4].isdigit()]
i = max(nums, default=0) + 1
print(f'มีรูป {len(nums)} รูปแล้ว เริ่มบันทึกที่ {name}/{i}.jpg  (กด s เพื่อถ่าย, q เพื่อออก)')
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
            if cv2.imwrite(f'{name}/{i}.jpg', face):
                print(f'บันทึก {name}/{i}.jpg')
                i += 1
            else:
                print(f'บันทึก {name}/{i}.jpg ไม่สำเร็จ')
        # q / Esc / กด X ปิดหน้าต่าง
        elif key in (ord('q'), 27) or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()