import cv2
import os
import sys
import numpy as np
import yaml

# 确保无论从项目根目录还是直接运行本脚本，都能导入 utils.logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("utils.calibration")

# 加载两个摄像头图片文件夹并将里面的彩图转换为灰度图
def load_images(folder, images):
    img_list = []
    for img_name in images:
        img_path = os.path.join(folder, img_name)
        
        frame = cv2.imread(img_path)
        # print(frame.shape)
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_list.append([frame, gray])
        else:
            logger.error(f"无法读取图像: {img_path}")
    return img_list

# 检测棋盘格角点
def get_corners(imgs, pattern_size):
    corners = []
    for frame, gray in imgs:
        ret, c = cv2.findChessboardCorners(gray, pattern_size)     #ret 表示是否成功找到棋盘格角点，c 是一个数组，包含了检测到的角点的坐标
        if not ret:
            logger.warning("未能检测到棋盘格角点")
            continue
        c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))     #cv2.cornerSubPix 函数用于提高棋盘格角点的精确度，对初始检测到的角点坐标 c 进行优化
        corners.append(c)      #将优化后的角点坐标 c 添加到 corners 列表中

        # 绘制角点并显示
        vis = frame.copy()
        cv2.drawChessboardCorners(vis, pattern_size, c, ret)
        # new_size = (1280, 800)
        # resized_img = cv2.resize(vis, new_size)
        # cv2.imshow('Corners', resized_img)
        # cv2.waitKey(150)

    return corners

# 相机标定
def calibrate_camera(object_points, corners, imgsize):
    cm_input = np.eye(3, dtype=np.float32)
    ret = cv2.calibrateCamera(object_points, corners, imgsize, cm_input, None)
    return ret

def save_calibration_to_yaml(file_path, cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T, E, F, imgsize, y_offset=0):
    data = {
        'imgsize': {
            'rows': 1,
            'cols': 2,
            'dt': 'd',
            'data': imgsize.flatten().tolist()
        },
        'y_offset': {
            'dt': 'd',
            'data': y_offset,
        },
        'camera_matrix_left': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': cameraMatrix_l.flatten().tolist()
        },
        'dist_coeff_left': {
            'rows': 1,
            'cols': 5,
            'dt': 'd',
            'data': distCoeffs_l.flatten().tolist()
        },
        'camera_matrix_right': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': cameraMatrix_r.flatten().tolist()
        },
        'dist_coeff_right': {
            'rows': 1,
            'cols': 5,
            'dt': 'd',
            'data': distCoeffs_r.flatten().tolist()
        },
        'R': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': R.flatten().tolist()
        },
        'T': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': T.flatten().tolist()
        },
        'E': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': E.flatten().tolist()
        },
        'F': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': F.flatten().tolist()
        }
    }

    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    logger.info(f"Calibration parameters saved to {file_path}")


def save_calibration1_to_yaml(file_path, cameraMatrix_l, distCoeffs_l, imgsize):
    data = {
        'imgsize': {
            'rows': 1,
            'cols': 2,
            'dt': 'd',
            'data': imgsize.flatten().tolist()
        },
        'camera_matrix_left': {
            'rows': 3,
            'cols': 3,
            'dt': 'd',
            'data': cameraMatrix_l.flatten().tolist()
        },
        'dist_coeff_left': {
            'rows': 1,
            'cols': 5,
            'dt': 'd',
            'data': distCoeffs_l.flatten().tolist()
        },
        
    }

    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    logger.info(f"Calibration parameters saved to {file_path}")


# 计算重投影误差
def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        total_points += len(imgpoints2)
    mean_error = total_error / total_points
    return mean_error

# 标定程序
def calibrate(left_folder = "./calibration/left",       # 左相机采集图像
              right_folder = "./calibration/right",     # 左相机采集图像
              pattern_size = (8, 6),                    # 棋盘格大小
              pattern_width = 17,                       # 棋盘格边长 (mm)
              y_offset=0,
              config_yaml_root = 'camera_config.yaml'):                      

    # 定义文件夹路径
    # 获取图像文件列表并排序
    left_images = sorted(os.listdir(left_folder))
    right_images = sorted(os.listdir(right_folder))

    # 确保左右相机图像数量一致
    assert len(left_images) == len(right_images), "左右相机图像数量不一致"

    img_left = load_images(left_folder, left_images)      #img_left是个列表，存放左摄像头所有的灰度图片。
    img_right = load_images(right_folder, right_images)

    if y_offset != 0:
        for i in range(len(img_left)):
            y1 = y_offset // 2
            y2 = y_offset - y1
            t1 = np.float32([[1, 0, 0],
                [0, 1, y1]])
            t2 = np.float32([[1, 0, 0],
                            [0, 1, -y2]])
            
            img_left[i][0] = cv2.warpAffine(img_left[i][0], t1, (img_left[i][0].shape[1], img_left[i][0].shape[0]), 
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            img_left[i][1] = cv2.warpAffine(img_left[i][1], t1, (img_left[i][1].shape[1], img_left[i][1].shape[0]), 
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            img_right[i][0] = cv2.warpAffine(img_right[i][0], t2, (img_right[i][0].shape[1], img_right[i][0].shape[0]), 
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            img_right[i][1] = cv2.warpAffine(img_right[i][1], t2, (img_right[i][1].shape[1], img_right[i][1].shape[0]), 
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            cv2.imshow('st', img_left[i][0] // 2 + img_right[i][0] // 2)
            cv2.imshow('st2', img_left[i][1] // 2 + img_right[i][1] // 2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # pattern_width = 16   # 棋盘格边长15mm，需要测量
    logger.info('开始识别角点...')
    corners_left = get_corners(img_left, pattern_size)       #corners_left的长度表示检测到棋盘格角点的图像数量。corners_left[i] 和 corners_right[i] 中存储了第 i 张图像检测到的棋盘格角点的二维坐标。
    corners_right = get_corners(img_right, pattern_size)
    cv2.destroyAllWindows()

    # 断言，确保所有图像都检测到角点
    assert len(corners_left) == len(img_left), "有图像未检测到左相机的角点"
    assert len(corners_right) == len(img_right), "有图像未检测到右相机的角点"

    # 准备标定所需数据
    points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)   #创建40 行 3 列的零矩阵，用于存储棋盘格的三维坐标点。棋盘格的大小是 8 行 5 列，40 个角点。数据类型为 np.float32，这是一张图的，因为一个角点对应一个三维坐标
    points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * pattern_width  #给这些点赋予实际的物理坐标，* 每个棋盘格的大小（mm）

    object_points = [points] * len(corners_left)     #包含了所有图像中棋盘格的三维物理坐标点 points。这里假设所有图像中棋盘格的物理坐标是相同的，因此用 points 复制 len(corners_left) 次。
    imgsize = img_left[0][1].shape[::-1]     #img_left[0] 是左相机图像列表中的第一张图像。img_left[0][1] 是该图像的灰度图像。shape[::-1] 取灰度图像的宽度和高度，并反转顺序，以符合 calibrateCamera 函数的要求。
    # print(imgsize)

    logger.info('开始左相机标定')
    ret_l = calibrate_camera(object_points, corners_left, imgsize)    #object_points表示标定板上检测到的棋盘格角点的三维坐标；corners_left[i]表示棋盘格角点在图像中的二维坐标；imgsize表示图像大小
    retval_l, cameraMatrix_l, distCoeffs_l, rvecs_l, tvecs_l = ret_l[:pattern_size[1]]    #返回值里就包含了标定的参数

    logger.info('开始右相机标定')
    ret_r = calibrate_camera(object_points, corners_right, imgsize)
    retval_r, cameraMatrix_r, distCoeffs_r, rvecs_r, tvecs_r = ret_r[:pattern_size[1]]

    # 立体标定，得到左右相机的外参：旋转矩阵、平移矩阵、本质矩阵、基本矩阵
    logger.info('开始立体标定')
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)
    ret_stereo = cv2.stereoCalibrate(object_points, corners_left, corners_right,
                                    cameraMatrix_l, distCoeffs_l,
                                    cameraMatrix_r, distCoeffs_r,
                                    imgsize, criteria=criteria_stereo,
                                    flags=cv2.CALIB_FIX_INTRINSIC)
    _, _, _, _, _, R, T, E, F = ret_stereo

    # 输出结果
    logger.info(f"左相机内参:\n{cameraMatrix_l}")
    logger.info(f"左相机畸变系数:\n{distCoeffs_l}")
    logger.info(f"右相机内参:\n{cameraMatrix_r}")
    logger.info(f"右相机畸变系数:\n{distCoeffs_r}")
    logger.info(f"旋转矩阵 R:\n{R}")
    logger.info(f"平移向量 T:\n{T}")
    logger.info(f"本质矩阵 E:\n{E}")
    logger.info(f"基本矩阵 F:\n{F}")
    logger.info("标定完成")

    # 保存标定结果
    save_calibration_to_yaml(config_yaml_root, cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T, E, F, np.array(list(imgsize)))
    logger.info(f'保存标定结果至{config_yaml_root}')
    # 计算并打印左相机和右相机的重投影误差
    logger.info(f"左相机重投影误差: {compute_reprojection_errors(object_points, corners_left, rvecs_l, tvecs_l, cameraMatrix_l, distCoeffs_l)}")
    logger.info(f"右相机重投影误差: {compute_reprojection_errors(object_points, corners_right, rvecs_r, tvecs_r, cameraMatrix_r, distCoeffs_r)}")

def calibrate1(folder,
               pattern_size = (8, 6),                    # 棋盘格大小
              pattern_width = 17,                       # 棋盘格边长 (mm)
              y_offset=0,
              config_yaml_root = 'camera_config_1.yaml' ):
    


    left_images = sorted(os.listdir(folder))

    img_left = load_images(folder, left_images)      #img_left是个列表，存放左摄像头所有的灰度图片。
    if y_offset != 0:
        for i in range(len(img_left)):
            y1 = y_offset
            # y2 = y_offset - y1
            t1 = np.float32([[1, 0, 0],
                [0, 1, y1]])
            # t2 = np.float32([[1, 0, 0],
            #                 [0, 1, -y2]])
            
            img_left[i][0] = cv2.warpAffine(img_left[i][0], t1, (img_left[i][0].shape[1], img_left[i][0].shape[0]), 
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            img_left[i][1] = cv2.warpAffine(img_left[i][1], t1, (img_left[i][1].shape[1], img_left[i][1].shape[0]), 
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # pattern_width = 16   # 棋盘格边长15mm，需要测量
    logger.info('开始识别角点...')
    corners_left = get_corners(img_left, pattern_size)       #corners_left的长度表示检测到棋盘格角点的图像数量。corners_left[i] 和 corners_right[i] 中存储了第 i 张图像检测到的棋盘格角点的二维坐标。
    cv2.destroyAllWindows()

    # 断言，确保所有图像都检测到角点
    assert len(corners_left) == len(img_left), "有图像未检测到左相机的角点"

    # 准备标定所需数据
    points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)   #创建40 行 3 列的零矩阵，用于存储棋盘格的三维坐标点。棋盘格的大小是 8 行 5 列，40 个角点。数据类型为 np.float32，这是一张图的，因为一个角点对应一个三维坐标
    points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * pattern_width  #给这些点赋予实际的物理坐标，* 每个棋盘格的大小（mm）

    object_points = [points] * len(corners_left)     #包含了所有图像中棋盘格的三维物理坐标点 points。这里假设所有图像中棋盘格的物理坐标是相同的，因此用 points 复制 len(corners_left) 次。
    imgsize = img_left[0][1].shape[::-1]     #img_left[0] 是左相机图像列表中的第一张图像。img_left[0][1] 是该图像的灰度图像。shape[::-1] 取灰度图像的宽度和高度，并反转顺序，以符合 calibrateCamera 函数的要求。
    logger.info(f'图像尺寸: {imgsize}')

    logger.info('开始相机标定')
    ret_l = calibrate_camera(object_points, corners_left, imgsize)    #object_points表示标定板上检测到的棋盘格角点的三维坐标；corners_left[i]表示棋盘格角点在图像中的二维坐标；imgsize表示图像大小
    retval_l, cameraMatrix_l, distCoeffs_l, rvecs_l, tvecs_l = ret_l[:pattern_size[1]]    #返回值里就包含了标定的参数

    # 输出结果
    logger.info(f"相机内参:\n{cameraMatrix_l}")
    logger.info(f"相机畸变系数:\n{distCoeffs_l}")

    # 保存标定结果
    save_calibration1_to_yaml(config_yaml_root, cameraMatrix_l, distCoeffs_l, np.array(list(imgsize)))
    logger.info(f'保存标定结果至{config_yaml_root}')
    # 计算并打印左相机和右相机的重投影误差
    logger.info(f"重投影误差: {compute_reprojection_errors(object_points, corners_left, rvecs_l, tvecs_l, cameraMatrix_l, distCoeffs_l)}")
    
if __name__ == '__main__':
    '''
        left_folder:        左视角存储路径
        right_folder:       右视角存储路径
        pattern_width:      标定板正方形边长/mm
        pattern_size:       网格数
        config_yaml_root:   标注结果保存地址
    '''
    calibrate(left_folder = "calibrations/calibration_dechuang1/left", 
              right_folder = "calibrations/calibration_dechuang1/right", 
              pattern_size = (11, 8),
              pattern_width = 25, 
            #   y_offset=-34,
              config_yaml_root = './model_data/configs/camera_config_dechuang_640x480_ttt.yaml')


    # calibrate1(folder = "calibrations/calibration_dechuang1/right", 
    #           pattern_size = (11, 8),
    #           pattern_width = 25, 
    #           y_offset=17,
    #           config_yaml_root = './model_data/configs/camera_config_dechuang_640x480_2.yaml')