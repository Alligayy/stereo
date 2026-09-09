from ultralytics import YOLO

# 加载预训练模型
model = YOLO("model_data/yolo/yolov8n-seg.pt")  # 或自定义训练模型路径

# 导出ONNX
model.export(
    format="onnx",
    opset=18,               # ONNX算子集版本(推荐12-17)
    imgsz=(640, 640),       # 输入尺寸(H,W)
    dynamic=False,           # 启用动态维度
    # simplify=True,          # 简化模型(需onnxsim)
    half=False,             # FP16量化(需GPU支持)
    # device="cpu"            # 导出设备(cpu/cuda)
)