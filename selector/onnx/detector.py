import numpy as np
from .transforms import Normalize, Compose
from onnxruntime import InferenceSession

CANVAS_SHAPE = (384, 384)


class BaseDetector:
    """ 调用onnx模型进行预测的基类 """

    def __init__(self, onnx_path):
        self.sess = InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.transform = Compose([Normalize()])

    def infer(self, image):
        """ 推理方法 """
        inputs = self.transform({'img': image})['img']
        inputs = inputs[np.newaxis, ...]
        inputs = inputs[..., :CANVAS_SHAPE[0], :CANVAS_SHAPE[1]]
        ort_outs = self.sess.run(output_names=None, input_feed={self.sess.get_inputs()[0].name: inputs})
        return ort_outs

    def postprocess(self, ort_outs):
        """ 后处理 """
        pass

    def __call__(self, image):
        ort_outs = self.infer(image)
        image = self.postprocess(ort_outs)
        return image


class NerveSegmenter(BaseDetector):
    """ 将原始图像分割为背景和神经 """

    def __init__(self, onnx_path):
        super().__init__(onnx_path)

    def postprocess(self, ort_outs):
        image = np.squeeze(ort_outs[0]) * 255
        image = image.astype('uint8')
        return image
