"""วัดความแม่นของ KNN ด้วยรูปที่มีอยู่ (ไม่ต้องเปิดกล้อง)

1. leave-one-out: หยิบรูปออกทีละ 1 รูป แล้วทายจากรูปที่เหลือทั้งหมด
2. แบ่งตามลำดับเวลา: เทรนด้วยรูป 70% แรกของแต่ละคน ทดสอบกับรูป 30% หลัง
   ใกล้เคียงการใช้งานจริงกว่า เพราะรูปที่ถ่ายติดกันหน้าเหมือนกันมาก leave-one-out จึงทายง่ายเกินจริง
"""
import os
import numpy as np
from FaceReg import load_faces

X, y = load_faces()
nums = []   # เลขรูป (1.jpg, 2.jpg, ...) ใช้เรียงตามลำดับที่ถ่าย
for name in sorted(set(y)):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.jpg'))
    nums += [int(f[:-4]) if f[:-4].isdigit() else 0 for f in files][:int((y == name).sum())]
nums = np.array(nums)

sq = (X ** 2).sum(1)
D = sq[:, None] + sq[None] - 2 * X @ X.T   # ระยะกำลังสองทุกคู่ คำนวณครั้งเดียว
np.fill_diagonal(D, np.inf)                # ห้ามทายจากรูปตัวเอง

test = np.zeros(len(y), bool)
for name in set(y):
    idx = np.flatnonzero(y == name)
    idx = idx[np.argsort(nums[idx])]
    test[idx[int(len(idx) * 0.7):]] = True


def vote(labels):
    cls, cnt = np.unique(labels, return_counts=True)
    return cls[np.argmax(cnt)]


print(f'{len(y)} รูป จาก {len(set(y))} คน\n')
print(f"{'k':>3}  {'leave-one-out':>16}  {'70/30 ตามลำดับเวลา':>20}")
for k in (1, 3, 5):
    nn = np.argsort(D, 1)[:, :k]
    loo = sum(vote(y[nn[i]]) == y[i] for i in range(len(y)))
    Dt = D[test][:, ~test]
    nt = np.argsort(Dt, 1)[:, :k]
    chrono = sum(vote(y[~test][nt[r]]) == y[i] for r, i in enumerate(np.flatnonzero(test)))
    print(f'{k:>3}  {loo:>4}/{len(y)} = {loo / len(y):6.1%}  {chrono:>4}/{test.sum()} = {chrono / test.sum():6.1%}')
