import math
import time
import cv2
import numpy as np
import onnxruntime as ort
import os
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
 
# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))
 
 
def nms(boxes, scores, iou_threshold):#极大值抑制
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
 
    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
 
        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
 
        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]
 
        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]
 
    return keep_boxes
 
 
def compute_iou(box, boxes): #计算IOU
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
 
    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
 
    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
 
    # Compute IoU
    iou = intersection_area / union_area
 
    return iou
 
 
def xywh2xyxy(x): #获取box
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
 
 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
 
def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None): #绘制检测 画框 写类别 写置信度
    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)
 
    mask_img = draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)
 
    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]
 
        x1, y1, x2, y2 = box.astype(int)
 
        # Draw rectangle
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)
 
        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)
 
        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
 
        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
 
    return mask_img
 
 
def draw_masks(image, boxes, class_ids, mask_alpha=0.3, mask_maps=None): #在框内绘制掩膜
    mask_img = image.copy()
 
    # Draw bounding boxes and labels of detections
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        color = colors[class_id]
 
        x1, y1, x2, y2 = box.astype(int)
 
        # Draw fill mask image
        if mask_maps is None:
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        else:
            crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
            crop_mask_img = mask_img[y1:y2, x1:x2]
            crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
            mask_img[y1:y2, x1:x2] = crop_mask_img
 
    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)
 
 
def draw_comparison(img1, img2, name1, name2, fontsize=2.6, text_thickness=3):
    (tw, th), _ = cv2.getTextSize(text=name1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img1.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img1, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (0, 115, 255), -1)
    cv2.putText(img1, name1,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)
 
    (tw, th), _ = cv2.getTextSize(text=name2, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img2.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img2, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (94, 23, 235), -1)
 
    cv2.putText(img2, name2,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)
 
    combined_img = cv2.hconcat([img1, img2])
    if combined_img.shape[1] > 3840:
        combined_img = cv2.resize(combined_img, (3840, 2160))
 
    return combined_img
 
class YOLOSeg:
 
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, num_masks=32):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks
 
        # Initialize model
        self.initialize_model(path)
 
    def __call__(self, image):
        return self.segment_objects(image)
 
    def initialize_model(self, path):

        print(os.path.dirname(path))
        trt_options = {
                'trt_fp16_enable': True,
                'trt_max_workspace_size': 6<<30,  # 6GB工作空间
                'trt_engine_cache_enable': True,    # 启用引擎缓存
                'trt_engine_cache_path': os.path.dirname(path),
                'trt_builder_optimization_level': 1,
                # 'trt_tactic_sources': '+CUBLAS,+CUDNN',
            }
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # self.session_options.intra_op_num_threads = 8
        self.session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(path,
                                                   sess_options=self.session_options, 
                                                    providers=[
                                                        ('TensorrtExecutionProvider', trt_options),
                                                            #    'CUDAExecutionProvider',
                                                            #    'CPUExecutionProvider'
                                                               ])#onnx模型的加载
 
        # self.session_options = ort.SessionOptions()
        # self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # # self.session_options.intra_op_num_threads = 8
        # self.session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        exit()
        # Get model info
        self.get_input_details()
        self.get_output_details()
 
    def segment_objects(self, image):
        input_tensor = self.prepare_input(image) #这是图像预处理 把图像处理成模型能接受的张量
 
        # Perform inference on the image   # 注意用dfl回归bbox的话，bbox的维度就是80+16*4+32
        outputs = self.inference(input_tensor)  # output[0] (1,4+80+32=116,80*80+40*40+20*20=8400)  //  output[1] (1,32,160,160)
        # 对输出0进行处理，返回bbox和对应的mask_feature
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])#模型的第一个输出 解析output0会得到检测框位置、置信度得分、对应的类别ID（0,1,2，...），以及mask_pred掩码的预测部分
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])#通过放缩后的图像得到掩码的预测部分、以及模型的输出output1将其掩码信息mask_maps转换到原图像上去
 
 
        return self.boxes, self.scores, self.class_ids, self.mask_maps
 
    def prepare_input(self, image): #将图片预处理
        self.img_height, self.img_width = image.shape[:2]#图像的原始尺寸
 
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#opencv读图格式默认是BGR 要转换成RGB
 
        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))#在初始化的时候 获取了model的输入尺寸 然后将图像resize成模型需要的尺寸
 
        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0 #归一化处理
        input_img = input_img.transpose(2, 0, 1)#这行代码改变了 input_img 的维度顺序。在图像处理中，图像通常以形状 (高度, 宽度, 通道数) 的数组形式存储（例如，在OpenCV中）。
        #然而，许多深度学习框架（包括使用ONNX Runtime的模型）期望输入数据的形状为 (通道数, 高度, 宽度)。因此，转置步骤是必要的，以确保数据维度与模型期望的输入格式相匹配。
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        # 首先将 input_img 转换为一个NumPy数组（如果它还不是的话），然后添加一个新的维度（在数组的最前面），这个新维度代表批处理大小（batch size）。在这个例子中，批处理大小为1，因为我们只处理一张图像。
        # 接着，它将数组的数据类型转换为 np.float32，这是许多深度学习模型所期望的数据类型。最终，input_tensor 是一个形状为 (1, 通道数, 高度, 宽度) 的四维张量，准备好被送入模型进行推理。
 
        return input_tensor
 
    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})#模型推理 将图像数据送入模型中
        # outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})：这是方法的核心部分，它执行模型的推理过程。
        # self.session：这通常指的是一个会话（session）对象，它是与模型进行交互的接口。其实就是加载的模型
        # self.output_names：这是一个列表或集合，包含了模型输出层的名称。在推理过程中，我们关注的是这些输出层的值。
        # {self.input_names[0]: input_tensor}：这是一个字典，它指定了模型的输入。self.input_names[0] 是模型输入层的名称，input_tensor 是传递给这个输入层的数据。
        # 这意味着方法接受一个输入张量（input_tensor），并将其传递给模型的输入层。
 
        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs
 
    def process_box_output(self, box_output):#output0的解析
 
        predictions = np.squeeze(box_output).T#这行代码首先使用 np.squeeze 函数去除 box_output 中所有单维度条目，即如果某个维度的大小为1，则这个维度会被移除。然后，使用 .T 对结果进行转置，以便后续处理。
        num_classes = box_output.shape[1] - self.num_masks - 4#这行代码计算目标类别的数量。它假设 box_output 的第二维（shape[1]）包含了所有输出（包括类别分数、边界框坐标、掩码等），
        #然后从中减去掩码的数量（self.num_masks）和4（通常代表边界框的坐标：x_min, y_min, x_max, y_max）。
 
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)#这行代码计算每个目标的最高类别置信度分数。它取 predictions 数组中每个目标（每行）的类别分数部分（从第5列开始，共 num_classes 列），并使用 np.max 函数找到每行的最大值，即每个目标的最高类别置信度。
        predictions = predictions[scores > self.conf_threshold, :]#这行代码过滤掉置信度低于某个阈值（self.conf_threshold）的目标。
        scores = scores[scores > self.conf_threshold]#类似地，这行代码也过滤掉低于置信度阈值的分数。
 
        if len(scores) == 0:
            return [], [], [], np.array([])
 
        box_predictions = predictions[..., :num_classes+4]#这行代码提取出包含边界框坐标和类别分数的部分。
        mask_predictions = predictions[..., num_classes+4:]#这行代码提取出包含掩码预测的部分。
 
        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)#这行代码找到每个目标最高置信度对应的类别ID。它取 box_predictions 中每个目标的类别分数部分，并使用 np.argmax 函数找到每行最大值的索引。
 
        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)#最后得到一个放缩回原始图像的boxes的坐标
 
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)#通过非极大值抑制去掉多余的检测框
 
        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices] #最后只留下极大值抑制以后的信息
 
    def process_mask_output(self, mask_predictions, mask_output):#定义一个方法process_mask_output，它接受mask_predictions（掩码预测）和mask_output（模型直接输出的掩码特征图）作为输入。
 
        if mask_predictions.shape[0] == 0:
            return []
 
        mask_output = np.squeeze(mask_output)#使用np.squeeze去除mask_output中所有单维度的条目，这通常用于去除批处理维度（如果只有一个样本）或多余的通道维度（如果掩码是单通道的）。
 
        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW 获取mask_output的形状，这里假设它是(N, H, W)，其中N是掩码的数量，H和W是掩码的高度和宽度。
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        #将mask_output重塑为(num_mask, -1)，以便与mask_predictions进行矩阵乘法。这里假设mask_predictions是一个分类器输出的权重矩阵。应用sigmoid函数将结果转换为概率值。
        masks = masks.reshape((-1, mask_height, mask_width))#将结果重塑回(N, H, W)的形状。
 
        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width))#将掩码信息映射回原图像 scale_boxes就是原图像的掩码信息
 
        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))#初始化一个全零的数组mask_maps，用于存储最终的掩码图。
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))#计算模糊大小blur_size，这是基于原始图像尺寸和掩码尺寸的比例。
 
        # 计算缩放后的边界框坐标（scale_x1, scale_y1, scale_x2, scale_y2）和原始边界框坐标（x1, y1, x2, y2）。
        # 从masks中提取对应于缩放后边界框的掩码部分（scale_crop_mask）。
        # 使用cv2.resize将提取的掩码部分调整到原始边界框的尺寸。
        # 使用cv2.blur对掩码进行模糊处理，以减少噪声。
        # 将模糊后的掩码二值化（大于0.5的值为1，否则为0），并存储到mask_maps中的正确位置。
        for i in range(len(scale_boxes)):
 
            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))
 
            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))
 
            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)
 
            crop_mask = cv2.blur(crop_mask, blur_size)
 
            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask
 
        return mask_maps
 
    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]#从预测输出中提取前四个值作为边界框的坐标。通常，这四个值代表边界框的中心点坐标（x, y）和宽度（w）、高度（h）。
 
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))
        #调用 self.rescale_boxes 方法来调整边界框的尺寸，使其与原始图像的尺寸相匹配。这里传入的参数包括边界框、模型的输入尺寸（input_height, input_width）和原始图像的尺寸（img_height, img_width）。
 
 
        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)#将中心点+宽高格式转换为左上角+右下角格式的辅助函数，将边界框从 xywh 格式转换为 xyxy 格式。
 
        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)
        #函数确保边界框的每个坐标值都在图像的范围内。如果某个坐标值超出了图像的范围，np.clip 会将其限制在指定的最小值和最大值之间（这里是 0 和图像的宽度/高度）。
        return boxes
 
    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)#mask_maps这个参数是 是否绘制掩码信息 如果不传 默认是None 不绘制
 
    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha, mask_maps=self.mask_maps)
 
    def get_input_details(self):
        model_inputs = self.session.get_inputs()#这里是直接获取onnx需要的输入尺寸 后续无论什么样的图 都可以自适应的resize成onnx需要的尺寸
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))] #获取模型的输入名字 通过https://netron.app/打开可以看到 输入名称叫image
        self.input_shape = model_inputs[0].shape
        self.input_height = 640#self.input_shape[2]
        self.input_width = 640#self.input_shape[3]

        print(self.input_shape)
 
    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]#获取模型的输出头的名称 分割是两个 一个是output0 一个是output1
 
    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])#创建一个数组，其中包含输入尺寸的宽度、高度、宽度、高度。这里的输入尺寸可能是模型的输入图像尺寸。
        boxes = np.divide(boxes, input_shape, dtype=np.float32)#boxes与input_shape进行除法操作
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])#在与原图像尺寸进行乘法操作，将xywh放缩回原始图像，获取原始图像对应的xywh
 
        return boxes
 
 
if __name__ == '__main__':
 
    # #model_path = r'D:\yolo_11\ultralytics-main\model\yolov8l-seg.ONNX'
    model_path = 'model_data/yolo/yolov8n-seg.onnx'
    # # Initialize YOLOv8 Instance Segmentator
    yoloseg = YOLOSeg(model_path, conf_thres=0.3, iou_thres=0.5)
    # img = cv2.imread('testfile/zed_left_1748247842765.png')
    # print(img.shape)
 
    # # Detect Objects
    # yoloseg(img)
 
    # # Draw detections
    # combined_img = yoloseg.draw_masks(img)#draw_detections函数为不绘制掩码信息，只绘制检测框信息 draw_masks函数为绘制掩码信息及绘制检测框信息
    # print(combined_img.shape)
    # exit()
    # cv2.imwrite("Output.jpg", combined_img)
    # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    # cv2.imshow("Output", combined_img)
    # cv2.waitKey(0)




    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps, t1 = 0, 0
    
    while True:
        # 读取图像
        ret, frame = cap.read()
    
        if not ret:
            print('相机未找到，停止')
            break

        # splity = frame.shape[1] // 2
        # frame1, frame2 = frame[:, :splity], frame[:, splity:]
        

        yoloseg(frame)
        img = yoloseg.draw_masks(frame)


        cv2.imshow('img', img)
        key = cv2.waitKey(1)
        if key == 27: # esc
            break



    cap.release()
    cv2.destroyAllWindows()