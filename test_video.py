import numpy as np
from core import Model
from utils.CameraConfig import CameraConfig
# import model_data.cameraconfig_video as camera_config
import cv2
import os
from utils.mask_process import draw_image, get_decodes
import imageio
import platform
import gc
import time
from utils.logger import get_logger

logger = get_logger("test_video")

def video_depth(input_video, model_name, yaml_path, detect=0, onnx=True):
    current_os = platform.system()
    logger.info(f"当前操作系统：{current_os}")
    # camera_config = CameraConfig(yaml_path)
    # import model_data.cameraconfig_video as camera_config
    logger.info(f"正在初始化模型: {model_name} ...")
    model = Model(model_name=model_name, camera_config=yaml_path, onnx=onnx)
    logger.info(f"模型 {model_name} 初始化完成")
    if detect:
        from ultralytics import YOLO
        logger.info("加载 YOLO 检测模型 ...")
        yolo = YOLO('./model_data/yolo/yolov8n-seg.pt')

    # dir_path = os.path.dirname(input_video)
    filename, ext = os.path.splitext(input_video)
    output_video_path = os.path.join(f'{filename}-{model_name}.mp4')
    logger.info(f"相机配置: {yaml_path} | onnx: {onnx} | detect: {detect}")
    logger.info(f"输入视频: {input_video}")
    logger.info(f"输出视频: {output_video_path}")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {input_video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"视频帧率: {fps}")
    # print(fps)
    # exit()
    
    if current_os == "Linux":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if current_os == "Windows":
        writer = imageio.get_writer(output_video_path, mode='I', fps=fps)

    frame_idx = 0
    while True:
        # 读取图像
        ret, frame = cap.read()
        if not ret:
            logger.info(f"视频读取结束，共处理 {frame_idx} 帧")
            break
        frame_idx += 1
        t0 = time.time()

        try:
            splity = frame.shape[1] // 2
            frame1, frame2 = frame[:, :splity], frame[:, splity:]

            # 畸变矫正
            res = model.update(frame1, frame2, rate=1, remap=True, show_text=False)
            imgL, imgR = res['image1'], res['image2']
            # model.get_disparity(imgL, imgR)
            # model.get_depthmap()

            if detect:
                results = yolo.predict(imgL, conf=0.6)[0]
                decodes = get_decodes(results, yolo.names, res['threeD'], model.camera_config.Q)
                imgL = draw_image(imgL, decodes, 'm', 2, 'whxyz')

            cv2.putText(imgL, 'left', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(imgR, 'right', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            img = np.hstack((imgL, res['disp_map']))

            if current_os == "Windows":
                writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if current_os == "Linux":
                out.write(img)

            cv2.imshow('img', img)
            # cv2.imshow('disp', model.disp_map)
        except Exception as e:
            logger.exception(f"处理第 {frame_idx} 帧时发生错误: {e}")
            break

        logger.debug(f"第 {frame_idx} 帧处理完成，耗时 {time.time() - t0:.3f}s")

        key = cv2.waitKey(1)
        if key == 27:  # esc
            logger.info("检测到 ESC 键，退出循环")
            break

        if key == 32:  # space
            logger.info("检测到空格键，显示点云")
            try:
                model.show_pcd(res)
            except Exception as e:
                logger.exception(f"显示点云失败: {e}")

    cap.release()
    cv2.destroyAllWindows()
    del model
    gc.collect()
    if current_os == "Linux":
        out.release()
    if current_os == "Windows":
        writer.close()
    logger.info(f"模型 {model_name} 处理结束，资源已释放")
    # out.release()



if __name__ == '__main__':
    
    # model_names = ['sgbm', 'psm', 'unimatch', 'raft', 'sea-raft', 'cre', 'foundation', 'monster']
    model_names = ['foundation']
    # yaml_path = 'model_data/configs/camera_config_dechuang_640x480.yaml'
    # input_video = 'testfile/videos/20250602133949.mp4'

    # yaml_path = 'model_data/configs/camera_config_car.yaml'
    # input_video = 'testfile/videos/car.avi'   

    yaml_path = 'model_data/configs/camera_config_DECXIN_640x480_skk.yaml'
    input_video = 'testfile/videos/stereo_20261025_161129.mp4'  # stereo_20261008_171220.mp4
    
    for model_name in model_names:
        try:
            video_depth(input_video, model_name, yaml_path, detect=0, onnx=False)
        except Exception as e:
            logger.exception(f"模型 {model_name} 运行失败: {e}")

