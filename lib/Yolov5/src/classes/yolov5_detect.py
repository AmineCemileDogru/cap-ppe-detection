
import torch
import numpy as np

from capsules.Yolov5.src.lib.yolov5.utils.augmentations import letterbox
from capsules.Yolov5.src.lib.yolov5.utils.general import LOGGER, Profile, check_img_size, non_max_suppression, scale_boxes
from capsules.Yolov5.src.lib.yolov5.utils.plots import Annotator, colors


class Yolov5Detect(object):
    def __init__(
            self,
            model,
            source,  # numpy array image or image list
            device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            classes=None,
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            update=False,  # update all models
            line_thickness=3,  # bounding box thickness (pixels)
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            dnn=False  # use OpenCV DNN for ONNX inference
    ):
        self.model = model
        self.source = source
        self.device = device
        self.classes = classes
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.line_thickness = line_thickness
        self.update = update
        self.dnn = dnn
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf

    def run(self):
        source_copy = np.copy(self.source)
        return self.inference(source_copy, self.model, s='image:')

    def inference(self, source, model, s):
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        bs = 1  # batch_size
        im0 = source  # original image
        # Preprocessing
        im = letterbox(source, imgsz, stride=stride, auto=pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
            pred = model(im, augment=self.augment, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(prediction=pred, classes=self.classes, conf_thres=self.conf_thres, iou_thres=self.iou_thres, agnostic=self.agnostic_nms, max_det=self.max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            annotator = Annotator(np.ascontiguousarray(im0), line_width=self.line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Annotate
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

        return pred, names, im0

    # def check_and_convert_onnx_opset_version(self, target_opset_version, target_path, e):
    #     import onnx
    #     from onnx import version_converter
    #     original_model = onnx.load(self.weights)
    #
    #     # Inspect the opset_import attribute of the model
    #     opset_imports = original_model.opset_import
    #
    #     # Extract the opset version from the opset_imports list
    #     opset_version = -1
    #     for opset_import in opset_imports:
    #         if opset_import.version > opset_version:
    #             opset_version = opset_import.version
    #
    #     if opset_version != target_opset_version:
    #         converted_model = version_converter.convert_version(original_model, target_version=target_opset_version)
    #         onnx.save(converted_model, target_path.replace(".onnx","") + "_converted.onnx")
    #     else:
    #         print(f"Error loading model: {e}")
