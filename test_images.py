import numpy as np
from core import Model
import cv2
import os
import time
import gc
from utils.mask_process import draw_image, get_decodes
from utils.logger import get_logger

logger = get_logger("test_images")


def save_results(res, left_path, save_dir):
    """展示结束后保存深度图与点云，文件名按左视图命名"""
    if res.get('threeD') is None or res.get('mask') is None:
        logger.warning("无三维/掩码数据（可能未提供有效标定 Q），跳过保存")
        return

    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(left_path))[0]
    mask = res['mask'].astype('bool')

    # 深度图：threeD 的 Z 通道，无效位置 0
    depth = np.where(mask, res['threeD'][:, :, 2], 0).astype('float32')
    depth_npy = os.path.join(save_dir, f'{base}-depth.npy')
    np.save(depth_npy, depth)
    # 深度图伪彩色可视化
    depth_vis = cv2.normalize(np.abs(depth), None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_png = os.path.join(save_dir, f'{base}-depth.png')
    cv2.imwrite(depth_png, cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO))
    logger.info(f"深度图已保存: {depth_npy} | {depth_png}")

    # 点云：与 model.show_pcd 一致（threeD/1000，颜色取左图）
    try:
        import open3d as o3d
        indices = np.where(mask)
        points = res['threeD'][indices].reshape(-1, 3) / 1000.0
        colors = cv2.cvtColor(res['image1'], cv2.COLOR_BGR2RGB)[indices].reshape(-1, 3) / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_path = os.path.join(save_dir, f'{base}-pcd.ply')
        o3d.io.write_point_cloud(pcd_path, pcd)
        logger.info(f"点云已保存: {pcd_path} | 点数: {len(points)}")
    except Exception as e:
        logger.exception(f"点云保存失败: {e}")


def image_depth(left_path, right_path, model_name, yaml_path, detect=0, onnx=True, save_dir=None):
    logger.info(f"正在初始化模型: {model_name} ...")
    model = Model(model_name=model_name, camera_config=yaml_path, onnx=onnx)
    logger.info(f"模型 {model_name} 初始化完成")

    if detect:
        from ultralytics import YOLO
        logger.info("加载 YOLO 检测模型 ...")
        yolo = YOLO('./model_data/yolo/yolov8n-seg.pt')

    logger.info(f"相机配置: {yaml_path} | onnx: {onnx} | detect: {detect}")
    logger.info(f"输入图像: {left_path} | {right_path}")

    # 读取图像
    imgL_raw = cv2.imread(left_path)
    imgR_raw = cv2.imread(right_path)
    if imgL_raw is None or imgR_raw is None:
        logger.error(f"无法读取输入图像: {left_path} | {right_path}")
        return
    logger.info(f"图像读取成功 | 左图尺寸: {imgL_raw.shape} | 右图尺寸: {imgR_raw.shape}")

    t0 = time.time()
    try:
        # 畸变矫正
        res = model.update(imgL_raw, imgR_raw, rate=1, remap=True, show_text=False)
        imgL, imgR = res['image1'], res['image2']

        if detect:
            results = yolo.predict(imgL, conf=0.6)[0]
            decodes = get_decodes(results, yolo.names, res['threeD'], model.camera_config.Q)
            imgL = draw_image(imgL, decodes, 'm', 2, 'whxyz')

        cv2.putText(imgL, 'left', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(imgR, 'right', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        img = np.hstack((imgL, res['disp_map']))
        cv2.imshow('img', img)
    except Exception as e:
        logger.exception(f"处理图像时发生错误: {e}")
        cv2.destroyAllWindows()
        return

    logger.info(f"处理完成，耗时 {time.time() - t0:.3f}s")

    while True:
        key = cv2.waitKey(0)
        if key == 27:  # esc
            logger.info("检测到 ESC 键，退出")
            break

        if key == 32:  # space
            logger.info("检测到空格键，显示点云")
            try:
                model.show_pcd(res)
            except Exception as e:
                logger.exception(f"显示点云失败: {e}")

    cv2.destroyAllWindows()

    # 展示结束后自动保存深度图与点云
    if save_dir:
        save_results(res, left_path, save_dir)

    del model
    gc.collect()
    logger.info(f"模型 {model_name} 处理结束，资源已释放")


if __name__ == '__main__':

    model_names = ['foundation']
    yaml_path = 'model_data/configs/camera_config_DECXIN_640x480_skk.yaml'
    # left_path = 'testfile/images/20261026_025447/20261026_025447_0_left.png'
    # right_path = 'testfile/images/20261026_025447/20261026_025447_0_right.png'

    # left_path = 'testfile/images/20261026_030934/20261026_030934_0_left.png'
    # right_path = 'testfile/images/20261026_030934/20261026_030934_0_right.png'

    # left_path = 'testfile/images/20261026_031636/20261026_031636_0_left.png'
    # right_path = 'testfile/images/20261026_031636/20261026_031636_0_right.png'

    left_path = 'testfile/images/20261026_031938/20261026_031938_0_left.png'
    right_path = 'testfile/images/20261026_031938/20261026_031938_0_right.png'

    save_dir = 'testfile/images'  # 结果保存目录

    for model_name in model_names:
        try:
            image_depth(left_path, right_path, model_name, yaml_path, detect=0, onnx=False, save_dir=save_dir)
        except Exception as e:
            logger.exception(f"模型 {model_name} 运行失败: {e}")
