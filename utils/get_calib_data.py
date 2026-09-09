import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("utils.get_calib_data")

id_image = 0 # 图片的ID
camera = cv2.VideoCapture(2)
# 找到棋盘格的标准
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*2)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# 棋盘格数
chessid = (11, 8)

left_save_root = './calibrations/calibration_dechuang2/left'
right_save_root = './calibrations/calibration_dechuang2/right'

while True:
    ret, frame = camera.read()
    # 这里的左右两个摄像头的图像是连在一起的，所以进行一下分割
    # 裁剪坐标为[y0:y1, x0:x1]
    splity = frame.shape[1] // 2
    frame1 = frame[:, :splity]
    frame2 = frame[:, splity:]
    logger.debug(f'帧尺寸: {frame.shape}')
    # 转换成灰度图 棋盘格识别需要时灰度图
    grayR = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格这个 用的是9*6的棋盘格 用其他的要修改一下
    retR, cornersR = cv2.findChessboardCorners(grayR , chessid , None)
    retL, cornersL = cv2.findChessboardCorners(grayL , chessid , None)
    cv2.imshow('imgR', frame1)
    cv2.imshow('imgL', frame2)
    # 如果找到了棋盘格就显示叫角点
    if (retR == True) & (retL == True):
        corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)

        # 画出角点
        cv2.drawChessboardCorners(grayR, chessid, corners2R, retR)
        cv2.drawChessboardCorners(grayL, chessid, corners2L, retL)
        cv2.imshow('VideoR', grayR)
        cv2.imshow('VideoL', grayL)

        if cv2.waitKey(0) & 0xFF == ord('s'):  # S 存储 C 取消存储
            logger.debug('S PRESSES')
            str_id_image = str(id_image)
            logger.info(f'Images {str_id_image} saved for right and left cameras')
            if not os.path.exists(left_save_root):
                os.makedirs(left_save_root)
            if not os.path.exists(right_save_root):
                os.makedirs(right_save_root)
            cv2.imwrite(f'{left_save_root}/left-{str_id_image}.png', frame1)
            cv2.imwrite(f'{right_save_root}/right-{str_id_image}.png', frame2)
            
            id_image = id_image + 1

        else:
            logger.debug('Images not saved')


    if cv2.waitKey(1) & 0xFF == 27:  # esc 结束当前程序
        break

# Release the Cameras
camera.release()
cv2.destroyAllWindows()
