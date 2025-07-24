
import os
import sys

sys.path.append('/opt/project/capsules/Yolov5/src/lib/yolov5')
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.model import BoundingBox
from sdks.novavision.src.helper.executor import Executor
from capsules.PpeDetection.src.utils.utils import load_models
from capsules.Yolov5.src.classes.yolov5_detect import Yolov5Detect
from capsules.PpeDetection.src.utils.response import build_response
from capsules.PpeDetection.src.models.PackageModel import PackageModel, Detection


class PpeDetection(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.device = self.request.get_param("ConfigDevice")
        self.conf_thres = self.request.get_param("ConfidentThreshold")
        self.iou_thres = self.request.get_param("IOUThreshold")
        self.select_device = bootstrap["device"]
        self.weight = self.bootstrap.get("model")
        self.select_device = self.bootstrap.get("device")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model = load_models(config=config)
        return model

    def output_result(self, output, names, img_uid):
        output = output[0].numpy()
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

    def ppe_inference(self):
        output, names, im = Yolov5Detect(
            model=self.weight,
            source=self.image.value,
            device=str(self.select_device),
            conf_thres=float(self.conf_thres),
            iou_thres=float(self.iou_thres)
        ).run()
        output_detection_list = self.output_result(output, names, self.image.uID)
        return output_detection_list

    def run(self):
        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        self.detection = self.ppe_inference()
        packageModel = build_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()