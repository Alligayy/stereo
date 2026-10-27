# 项目简介

本项目是一个双目立体匹配算法集合库，提供的内容包括立体匹配、目标检测、双目相机的标定。

其中立体匹配包含如下算法：

| 模型名称    | 说明     |  原项目地址 |论文地址|
|------------|---------|-------|-------|
|sgbm |  经典立体匹配算法，基于opencv实现  | 无 ||
|psm |  2016年，精度较高，由于年份比较早，有很多改版，这里选用了比较稳定的pytorch版本  |[https://github.com/JiaRenChang/PSMNet](https://github.com/JiaRenChang/PSMNet) ||
|unimatch |  2023年，统一光流、立体匹配和深度估计的多任务Transformer模型（CVPR后续工作）  |[https://github.com/autonomousvision/unimatch](https://github.com/autonomousvision/unimatch)||
|raftstereo |  2021年，基于RAFT光流框架的迭代优化立体匹配网络   |[https://github.com/princeton-vl/RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) ||
|crestereo | 2022年，级联循环网络，自适应群相关层处理非理想校正，Middlebury和ETH3D排名第一 |官方实现（MegEngine）：<br>[https://github.com/megvii-research/CREStereo](https://github.com/megvii-research/CREStereo)<br> pytorch实现 <br> [https://github.com/ibaiGorordo/CREStereo-Pytorch](https://github.com/ibaiGorordo/CREStereo-Pytorch) ||
|sea-raft | 2022年，RAFT光流算法的改进版本，通过引入拉普拉斯混合损失和刚性运动预训练，这里只取其x方向光流作为视差 |[https://github.com/princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)||
|foundationstereo |2025年，英伟达开源的零样本泛化立体匹配基础模型，百万级合成数据训练，CVPR满分论文 |[https://github.com/NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo) ||
|monster |  2025年，英特尔开源的双分支模型，融合单目深度与立体匹配，五榜第一  |[https://github.com/Junda24/MonSter](https://github.com/Junda24/MonSter)||
|DEFOM Stereo|2025年，KITTI/Middlebury/ETH3D零样本泛化SOTA|[https://github.com/Insta360-Research-Team/DEFOM-Stereo](https://github.com/Insta360-Research-Team/DEFOM-Stereo)||



# 使用流程
## 一、配置环境

### 1.1 Anaconda 和 Pytorch 安装
安装 Anaconda 和 VScode

选择 2.5.0 版本的 torch

python版本要求 >= 3.9 

（以上版本可以变更，一般不会影响程序运行。唯一可能出现的问题是在linux下较高版本的torch导出的onnx可能会有问题，其余完全不影响）

### 1.2 基础必要的库

需要安装以下库（pip），
直接运行
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

以上库是运行的必要库。如果后续有调优和加速需求可以参考 **1.3** ～ **1.5** 没有可以自动跳过


### 1.3 sgbm 的 wls 滤波
在使用 sgbm 模型时可能需要 wls 滤波，该方法基于 `opencv-contrib-python` 包，而安装 `ultralytics` 时会自动安装 `opencv-python`，因此安装完 `ultralytics` 后需要**立即**卸载 `opencv-python`，然后再安装 `opencv-contrib-python` ，否则会无法使用。
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-contrib-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果需要 raft-stereo 在 ubuntu 下的 c++ cuda 加速（仅限linux系统）

如果遇到 opencv 窗口无法启动的情况，可以尝试 `pip uninstall opencv-python-headless opencv-python` 然后重新安装 `opencv-python`

### 1.4 Raft-stereo 的采样模块 c++ 加速（realtime模式）
需要 c++ 加速:
#### ubuntu

安装 build-essential
```bash
sudo apt update
sudo apt install build-essential
```
- ---


### 1.5 Foundation-stereo 加速
`flash-attn` 非必须安装，可以提供注意力机制加速（仅在foundation stereo中使用），如果需要安装 `flash-attn` 要 c++ 依赖

- ubuntu

```bash
sudo apt update
sudo apt install build-essential
pip install flash-attn
```



**注意 foundation stereo 暂时不支持 cpu 版本的 torch**

xformers 也可以提供加速，可以按照自己的版本安装 xformers ，这里我的 cuda 版本是12.1 就按照如下命令安装，其余的可以替换 `cu121`  至你的版本即可（如果报错根据提供的可选版本号选择不超过你的cuda版本的安装即可）

如果被强制更新了 pytorch 可以在 [xformers github官网](https://github.com/facebookresearch/xformers) 下载对应版本的包。

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U xformers --index-url https://download.pytorch.org/whl/cu121
```



### 1.6 onnx+tensorrt 加速推理

安装 onnxruntime-gpu
```bash
pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果希望使用 onnxruntime 在 TensorrtExecutionProvider 模式下加速推理需要安装 tensorrt 可以参考：

[Ubuntu TensorRT安装 https://blog.csdn.net/ytusdc/article/details/140143354](https://blog.csdn.net/ytusdc/article/details/140143354)


[windows TensorRT安装 https://blog.csdn.net/qq_43406895/article/details/128635174](https://blog.csdn.net/qq_43406895/article/details/128635174)



tensorrt 下载页面

[https://developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)


安装满足版本要求的 tensorrt
检查 onnxruntime 是否支持 tensorrt：

```python
import tensorrt as trt
print(trt.__version__)

import onnxruntime as ort

# 检查可用的提供程序
providers = ort.get_available_providers()
print("可用的执行提供程序:", providers)

# 检查是否包含TensorRT
if 'TensorrtExecutionProvider' in providers:
    print("TensorRT执行提供程序可用。")
else:
    print("TensorRT执行提供程序不可用。请检查安装。")
```
如果输出 *TensorRT执行提供程序可用*   则证明安装成功


## 二、标定（如果需要三维重建，必须，如果仅需要视差图可以跳过）
### 2.1 获得标定数据
如果你自己有标定参数，跳到 **2.3**

需要一个双目摄像头并能获取图像，打开 `./utils/get_calib_data.py` 
选择合适的摄像头参数，括号填入数字选择摄像头
```python
camera = cv2.VideoCapture(0)
```
设置图像分辨率：
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```
注意如果是一体式双目摄像头图像宽要设置为单张图像宽的2倍。

如果相机无法用 usb 连接，则在对应位置修改为图像接口即可（ `np.ndarray` 类型， frame1,frame2）,下面同理
```python
while True:
    ret, frame = camera.read()
    # 这里的左右两个摄像头的图像是连在一起的，所以进行一下分割
    # 裁剪坐标为[y0:y1, x0:x1]
    splity = frame.shape[1] // 2
    frame1 = frame[:, :splity]
    frame2 = frame[:, splity:]
```

准备好标定板，并得到棋盘格的内角点个数，以 $8\times6$ 为例，对应的是 $9\times7$ 的网格，修改
```python
chessid = (8, 6)
```

修改左右相机视角标定数据保存路径
```python
left_save_root = './calibration/left'
right_save_root = './calibration/right'
```

运行 `getdata.py` 当相机检测到角点后会弹出绘制角点的新窗口，保存按 `S` ，不保存按 `C`，得到大约十几张标定数据后按 `Esc` 结束运行。注意采集的棋盘格尽量分布在视角各个区域，减小误差。

### 2.2 得到标定结果

打开 `./utils/calibration.py` ，修改最后的函数参数


```python
if __name__ == '__main__':
    '''
        left_folder:        左视角存储路径
        right_folder:       右视角存储路径
        pattern_width:      标定板正方形边长/mm
        pattern_size:       网格数
        config_yaml_root:   标注结果保存地址
    '''
    calibrate(left_folder = "./calibration/left", 
              right_folder = "./calibration/right", 
              pattern_size = (11, 8),
              pattern_width = 25, 
              config_yaml_root = './model_data/camera_config.yaml')

```
运行即可得到标定结果文件 `./model_data/camera_config.yaml` 

### 2.3 修改添加自己的标定参数
如果你自己有一套标定参数，那么可以新建一个 yaml 文件按照其内部格式修改：

```yaml
R:                      # 旋转矩阵
...
T:                      # 平移矩阵
...
camera_matrix_left:     # 左相机内参
...
camera_matrix_right:    # 右相机内参
...
dist_coeff_left:        # 左相机畸变（k1,k2,p1,p2,k3）
...
dist_coeff_right:       # 右相机畸变
...
imgsize:                # 图像尺寸
...

```
后续三位重建需要上面的标定参数。


## 三、项目运行

### 3.1 显示视差图
运行 `test_disparity.py` 或者  可以测试单独两张图像生成视差图。
设置 `model_name` 有以下模型可供选择：

`'sgbm', 'psm', 'raft', 'cre', 'sea-raft', 'unimatch', 'foundation', 'monster', 'defom', 'waft', 's2m2` 

如果在使用例如 foundationstereo 或者 monster 这样参数量大的模型遇到显存不足的问题，可以调节 

```python
res = model.update(frame1, frame2, rate=1, remap=True)
```

中 rate 参数，0～1 表示对图像的缩放比例，越小越省显存但是会降低精度。

update 函数输出是一个字典，包含所有可能需要的结果，有些结果在参数设置不完全的情况下会为 None

```py
res = {
        'disparity' : None,     # 视差 H*W
        'threeD' : None,        # 点云 H*W*3
        'disp_gray' : None,     # 归一化黑白深度图 H*W
        'disp_map' : None,      # 归一化彩色深度图 H*W
        'image1' : None,        # 畸变校正左视角图 H*W*3
        'image2' : None,        # 畸变校正右视角图 H*W*3
        'image1_org' : None,    # 左视角原始图 H*W*3
        'image2_org' : None,    # 右视角原始图 H*W*3
        }
```

可以在模型初始化中设置参数，具体可以看单独的模型文件参数

例如设置 raft stereo 的运行参数：
```py
model = Model(model_name='raft', 
              params={'mode':'realtime', 'valid_iters':7, 'corr_implementation':'reg_cuda'}，
              camera_config='model_data/camera_config_dechuang_640x480.yaml',)
```
`camera_config` 参数指的是相机内参，如果包含会进行畸变矫正和三维重建，否则只能输出视差图。


指定 sgbm 的参数文件：

```python
model = Model(model_name='sgbm', yaml_path='model_data/sgbm.yaml')
```

这里可以调节 SGBM 模型的各种参数

本项目提供了7种转换好的 onnx 模型，除了 unimatch 和 sgbm 均可以转换
如果有参数调整要重新输出 onnx 可以执行
`utils/make_onnx.py`

该脚本默认导出

在其中选定模型的名称即可，输出路径使用默认路径。
完成上述步骤后，可以在 model 初始化时指定使用onnx模式推理，例如

```python
model = Model(model_name='foundation', 
              camera_config='model_data/camera_config_dechuang_640x480.yaml',
              onnx=True,
              trt_flag=True,
              )
```
如果使用 tensorrt 第一次构建引擎需要较长时间，要耐心等待，速度能看到显著提升。


如果遇到内存不足的情况，可以尝试增加交换空间来实现 **（建议这一步提前运行！）**

Linux:

```bash
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

恢复之前的交换空间
```bash
sudo swapoff /swapfile
sudo rm /swapfile
```



### 3.2 深度目标测距+显示点云

如果有标定参数可以运行 `test_file.py` 显示三维点云。
标定参数在如下位置设置：
```python
camera_config = CameraConfig('model_data/camera_config.yaml')
```
注意上述默认的参数并不通用！！！如果是你自己的相机需要单独修改！！！

设置 `YOLO_DETETCT = True` 表示是否进行 yolo 目标检测
```python
if YOLO_DETETCT:
    from ultralytics import YOLO
    from utils.mask_process import draw_image, get_decodes
    yolo = YOLO('./model_data/yolov8-seg-face.pt')
```
修改 `'./model_data/yolov8-seg-face.pt'` 为你的模型的路径


获得检测信息

在代码中
```python
if YOLO_DETETCT:
    results = yolo.predict(imgL, conf=0.4)[0]
    decodes = get_decodes(results, yolo.names, model)
    imgL_board  = draw_image(imgL, decodes, mode='whxyz') 

```
其中 `get_decodes` 函数还有一个输入参数 `center=True`，表示是否使用标注框中心点的世界坐标作为输出结果，如果设置为 `False` 则表示使用滤波模型（见 **3.5**）

`decodes` 表示一个列表，包含若干个字典，其中每个字典的格式如下，供取用：

```python
dict1 = {
    # 标注框信息
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
    # 掩码矩阵（如果使用yolo-seg）
        'mask': mask_mat, 
        'class_name':class_name,
    # 世界坐标xyz
        'x':x,
        'y':y,
        'z':z,
        'mask_flag':mask_flag,
    # 宽高
        'width':width,
        'height':height,
    # 颜色
        'color':colors[class_id]，
        }

```
mode 表示显示内容，此处为 'whxyz' 表示显示宽高+世界坐标xyz

按 `Space` 显示点云，按 `Esc` 退出

### 3.3 相机实时检测
运行 `test_cam.py` 可以看到实时显示结果
关于摄像头参数设定参考 **2.1**

### 3.4 测试双目视频
运行 `test_video.py`，可以修改自己的视频路径，本例提供一个测试视频和与其对应的相机内参检测，保存结果为左视角图像和深度图的拼接结果。 
