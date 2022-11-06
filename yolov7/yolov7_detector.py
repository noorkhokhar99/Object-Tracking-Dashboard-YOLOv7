from .yolov7_utils import prepare_input, process_output
import onnxruntime
import os


class YOLOv7Detector:
    def __init__(self,
                 weights=os.path.join(os.path.dirname(
                     os.path.abspath(__file__)), './weights/yolov7.onnx'),
                 use_cuda=True, use_onnx=True) -> None:

        if use_onnx:
            if use_cuda:
                providers = [
                    'CUDAExecutionProvider',
                    'CPUExecutionProvider'
                ]
            else:
                providers = ['CPUExecutionProvider']

        self.model = onnxruntime.InferenceSession(weights, providers=providers)

        self.class_names= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'] 

        # else:
        #     self.model = torch
        self.device = 'cuda' if use_cuda else 'cpu'

    def detect(self, image: list,
               conf_thres: float = 0.25,
               iou_thres: float = 0.45,
               classes: list = None,
               agnostic_nms: bool = False,
               input_shape=(640, 640),
               max_det: int = 1000) -> list:

        image0 = image.copy()
        input_tensor = prepare_input(image, input_shape)
        input_name = self.model.get_inputs()[0].name

        outputs = self.model.run([self.model.get_outputs()[0].name], {
                                 input_name: input_tensor})
        dets = process_output(
            outputs, image0.shape[:2], input_shape, conf_thres, iou_thres, classes=classes)
        image_info = {
            'width': image0.shape[1],
            'height': image0.shape[0],
        }
        return dets, image_info
