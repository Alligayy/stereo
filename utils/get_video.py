import cv2
import os
import sys
import platform
from datetime import datetime
import imageio
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("utils.get_video")

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
save_root = './testfile/videos'
fps = 20

os.makedirs(save_root, exist_ok=True)
now = datetime.now() 
timestamp = now.strftime('%Y%m%d%H%M%S') 
current_os = platform.system()
output_video_path = os.path.join(save_root, f'{timestamp}.mp4')

current_os = platform.system()
start_flag = False
while True:
    ret, frame = camera.read()
    # 这里的左右两个摄像头的图像是连在一起的，所以进行一下分割
    # 裁剪坐标为[y0:y1, x0:x1]
    splity = frame.shape[1] // 2
    frame1 = frame[:, :splity]
    frame2 = frame[:, splity:]

    cv2.imshow('imgL', frame1)
    cv2.imshow('imgR', frame2)

    key = cv2.waitKey(1) & 0xFF

    if start_flag:
        img = np.hstack((frame1, frame2))
        if current_os == "Windows":
            writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if current_os == "Linux":
            out.write(img)

    if key == ord('s'):
        if current_os == "Linux":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width  = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
            height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if current_os == "Windows":
            writer = imageio.get_writer(output_video_path, mode='I', fps=fps)
        logger.info('开始录制视频')
        start_flag = True

    if key == 27:  # esc 结束当前程序
        logger.info('结束录制视频')
        break

if current_os == "Linux":
    out.release()
if current_os == "Windows":
    writer.close()
cv2.destroyAllWindows()
