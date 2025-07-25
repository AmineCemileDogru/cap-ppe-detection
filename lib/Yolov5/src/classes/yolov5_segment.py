

import numpy as np
import torch


from  capsules.Yolov5.src.lib.yolov5.models.common import DetectMultiBackend
from  capsules.Yolov5.src.lib.yolov5.utils.augmentations import letterbox
from  capsules.Yolov5.src.lib.yolov5.utils.general import (LOGGER, Profile, check_img_size, non_max_suppression,
                                                         scale_boxes,scale_segments, strip_optimizer)
from  capsules.Yolov5.src.lib.yolov5.utils.plots import Annotator, colors
from  capsules.Yolov5.src.lib.yolov5.utils.torch_utils import select_device
from  capsules.Yolov5.src.lib.yolov5.utils.segment.general import masks2segments, process_mask, process_mask_native


class Yolov5Segment(object):
    def __init__(self,
                 model,  # model.pt path(s)
                 source,  # file/dir/URL/glob/screen/0(webcam)
                 imgsz=(640, 640),  # inference size (height, width)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 view_img=False,  # show results
                 save_txt=False,  # save results to *.txt
                 save_conf=False,  # save confidences in --save-txt labels
                 save_crop=False,  # save cropped prediction boxes
                 nosave=False,  # do not save images/videos
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 visualize=False,  # visualize features
                 update=False,  # update all models
                 #project / 'runs/predict-seg',  # save results to project/name
                 name='exp',  # save results to project/name
                 exist_ok=False,  # existing project/name ok, do not increment
                 line_thickness=3,  # bounding box thickness (pixels)
                 hide_labels=False,  # hide labels
                 hide_conf=False,  # hide confidences
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 vid_stride=1,  # video frame-rate stride
                 retina_masks=False,
                 ):
        self.model = model
        self.source = source
        self.view_img=view_img
        self.save_txt=save_txt
        self.save_conf=save_conf
        self.save_crop=save_crop
        self.nosave=nosave
        self.visualize=visualize
        self.name=name
        self.exist_ok=exist_ok
        self.vid_stride=vid_stride
        self.retina_masks=retina_masks
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.classes = classes
        self.line_thickness = line_thickness
        self.update = update
        self.dnn = dnn
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf

    def run(self):
        source_copy = np.copy(self.source)
        return self.inference(source_copy, self.model, s = 'image:')

    def inference(self, source, model, s):
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        bs = 1  # batch_size
        im0 = source # original image
        # Preprocessing
        im = letterbox(source, imgsz, stride=stride, auto=pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
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
            pred, proto  = model(im, augment=self.augment, visualize=False)[:2]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det,nm=32)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            s += '%gx%g ' % im.shape[2:]  # print string
            #annotator = Annotator(np.ascontiguousarray(im0), line_width=self.line_thickness, example=str(names))
            if len(det):
                if self.retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                else:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Mask plotting
                #  annotator.masks(
                #    masks,
                #    colors=[colors(x, True) for x in det[:, 5]],
                #    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(2, 0, 1).flip(0).contiguous() /
                #          255 if self.retina_masks else im[i])"""
                labels=[]
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.3f}')
                        mask = masks[j].cpu().numpy()
                        labels.append({"name":names[c], "index":c, "confident":f'{conf:.3f}',"x":int(xyxy[0]),"y":int(xyxy[1]),"w":int(xyxy[2])-int(xyxy[0]),"h":int(xyxy[3])-int(xyxy[1]),"mask": mask})
                        #annotator.box_label(xyxy, label, color=colors(c, True))
                        #annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
            #im0 = annotator.result()

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

        # If nothing is found, labels are not created. Create Labels.
        if 'labels' in locals():
            return labels, source
        else:
            labels = []

        return labels, source
