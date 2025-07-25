
import os
import sys

sys.path.append('/opt/project/capsules/Yolov5/src/lib/yolov5')
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from capsules.Yolov5.src.utils.utils import ModelLoader
from sdks.novavision.src.helper.executor import Executor
from capsules.Yolov5.src.classes.yolov5_classify import Yolov5Classify
from capsules.Yolov5.src.utils.response import build_response_classify
from capsules.Yolov5.src.models.PackageModel import PackageModel, Detection


class Classify(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.conf_weights = self.request.get_param("Weights")
        self.device = self.request.get_param("ConfigDevice")
        self.file_extension_list = [".onnx", ".pt"]
        self.select_device = self.bootstrap["device"]
        self.model_accuracy = int(self.request.get_param("ConfigModelAccuracy"))
        self.image = self.request.get_param("inputImage")
        self.weight = self.bootstrap.get("model")
        self.select_device = self.bootstrap.get("device")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model = ModelLoader(config=config).load_model()
        return model

    def output_result(self, outputs, img_uid):
        detections = []
        for output in outputs:
            newdetect = Detection(confidence=output["confidence"], classLabel=output["name"], classId=output["index"],
                                  imgUID=img_uid)
            detections.append(newdetect)
        return detections

    def classification_inference(self):
        outputs, im = Yolov5Classify(
            model=self.weight,
            source=self.image.value,
            device=str(self.select_device),
            accuracy_level=self.model_accuracy
        ).run()
        detections = self.output_result(outputs, self.image.uID)
        return detections

    def run(self):
        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        self.detections = self.classification_inference()
        packageModel = build_response_classify(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
