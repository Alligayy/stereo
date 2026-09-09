import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("utils.get_img")

id_image = 0 # 图片的ID
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)
left_save_root = './test/left'
right_save_root = './test/right'

if not os.path.exists(left_save_root):
    os.makedirs(left_save_root)
if not os.path.exists(right_save_root):
    os.makedirs(right_save_root)

while True:
    ret, frame = camera.read()
    # 这里的左右两个摄像头的图像是连在一起的，所以进行一下分割
    # 裁剪坐标为[y0:y1, x0:x1]
    splity = frame.shape[1] // 2
    frame1 = frame[:, :splity]
    frame2 = frame[:, splity:]
    img_st = frame1 // 2 + frame2 // 2
    cv2.imshow('img_st', img_st)
    cv2.imshow('imgL', frame1)
    cv2.imshow('imgR', frame2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # S 存储 C 取消存储
        logger.debug('S PRESSES')
        str_id_image = str(id_image)
        logger.info(f'Images {str_id_image} saved for right and left cameras')
        cv2.imwrite(f'{left_save_root}/left-{str_id_image}.png', frame1)
        cv2.imwrite(f'{right_save_root}/right-{str_id_image}.png', frame2)
        logger.debug(f'图像尺寸: {frame1.shape}')
        id_image = id_image + 1


    if key == 27:  # esc 结束当前程序
        break

# Release the Cameras
# camera.release()
cv2.destroyAllWindows()
