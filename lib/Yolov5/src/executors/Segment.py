
import os
import sys
import json
import numpy as np
from imantics import Mask

sys.path.append('/opt/project/capsules/Yolov5/src/lib/yolov5')
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.response import Response
from sdks.novavision.src.base.model import BoundingBox
from capsules.Yolov5.src.utils.utils import ModelLoader
from sdks.novavision.src.helper.executor import Executor
from capsules.Yolov5.src.classes.yolov5_segment import Yolov5Segment
from capsules.Yolov5.src.utils.response import build_response_segment
from capsules.Yolov5.src.models.PackageModel import PackageModel, Detection, KeyPoints


class Segment(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.conf_weights = self.request.get_param("Weights")
        self.device = self.request.get_param("ConfigDevice")
        self.image = self.request.get_param("inputImage")
        self.classes = self.request.get_param("Classes")
        self.conf_thres = self.request.get_param("ConfidentThreshold")
        self.iou_threes = self.request.get_param("IOUThreshold")
        self.weight = self.bootstrap.get("model")
        self.select_device = self.bootstrap.get("device")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model = ModelLoader(config=config).load_model()
        return model

    def output_result(self, outputs, img_uid):
        detects = []
        for output in outputs:
            bbox = BoundingBox(left=output["x"], top=output["y"], width=output["w"], height=output["h"])
            if "mask" in output and isinstance(output["mask"], np.ndarray):
                mask = output["mask"]
                polygons = Mask(mask).polygons()
                key_points = [KeyPoints(cx=int(x), cy=int(y)) for x, y in polygons.points[0]]
            else:
                key_points = []

            newdetect = Detection(
                boundingBox=bbox,
                confidence=output["confident"],
                classLabel=output["name"],
                classId=output["index"],
                imgUID=img_uid,
                keyPoints=key_points
            )
            detects.append(newdetect)

        return detects

    def segment_inference(self):
        outputs, im = Yolov5Segment(
            model=self.weight,
            source=self.image.value,
            device=str(self.select_device),
            conf_thres=float(self.conf_thres),
            iou_thres=float(self.iou_threes),
            retina_masks=True
        ).run()
        detections = self.output_result(outputs, self.image.uID)
        return detections

    def run(self):
        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        self.detections = self.segment_inference()
        packageModel = build_response_segment(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()

