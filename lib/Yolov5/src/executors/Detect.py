
import os
import sys

sys.path.append('/opt/project/capsules/Yolov5/src/lib/yolov5')
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.model import BoundingBox
from capsules.Yolov5.src.utils.utils import ModelLoader
from sdks.novavision.src.helper.executor import Executor
from capsules.Yolov5.src.classes.yolov5_detect import Yolov5Detect
from capsules.Yolov5.src.utils.response import build_response_detect
from capsules.Yolov5.src.models.PackageModel import PackageModel, Detection


class Detect(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.conf_weights = self.request.get_param("Weights")
        self.device = self.request.get_param("ConfigDevice")
        self.conf_thres = self.request.get_param("ConfidentThreshold")
        self.iou_threes = self.request.get_param("IOUThreshold")
        self.image = self.request.get_param("inputImage")
        self.weight = self.bootstrap.get("model")
        self.select_device = self.bootstrap.get("device")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model = ModelLoader(config=config).load_model()
        return model

    def output_result(self, output, names, img_uid):
        output = output[0].cpu().numpy()
        detection_list = []
        for i in range(0, len(output)):
            bbox = BoundingBox(
                left=output[i][0],
                top=output[i][1],
                width=output[i][2] - output[i][0],
                height=output[i][3] - output[i][1]
            )
            newdetect = Detection(
                boundingBox=bbox,
                confidence=output[i][4],
                classLabel=names[(int(output[i][5]))],
                classId=int(output[i][5]),
                imgUID=img_uid
            )
            detection_list.append(newdetect)
        return detection_list

    def detection_inference(self, img):
        output, names, im = Yolov5Detect(
            model=self.weight,
            source=img.value,
            device=str(self.select_device),
            conf_thres=float(self.conf_thres),
            iou_thres=float(self.iou_threes)
        ).run()
        detections = self.output_result(output, names, img.uID)
        return detections

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        self.detections = self.detection_inference(img)
        packageModel = build_response_detect(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()