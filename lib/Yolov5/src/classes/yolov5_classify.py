import torch
import numpy as np
import torch.nn.functional as F

from capsules.Yolov5.src.lib.yolov5.utils.plots import Annotator, colors
from capsules.Yolov5.src.lib.yolov5.utils.augmentations import classify_transforms
from capsules.Yolov5.src.lib.yolov5.utils.general import LOGGER, Profile, check_img_size


class Yolov5Classify(object):
    def __init__(self,
                 model,  # model path or triton URL
                 source,  # numpy array image or image list
                 device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 imgsz=(224, 224),  # inference size (height, width)
                 view_img=False,  # show results
                 save_txt=False,  # save results to *.txt
                 nosave=False,  # do not save images/videos
                 augment=False,  # augmented inference
                 visualize=False,  # visualize features
                 update=False,  # update all models
                 name='exp',  # save results to project/name
                 exist_ok=False,  # existing project/name ok, do not increment
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 vid_stride=1,  # video frame-rate stride
                 accuracy_level=1
                 ):
        self.model = model
        self.source = source
        self.imgsz = imgsz
        self.view_img = view_img
        self.save_txt = save_txt
        self.nosave = nosave
        self.device = device
        self.visualize = visualize
        self.update = update
        self.dnn = dnn
        self.augment = augment
        self.name = name
        self.exist_ok = exist_ok
        self.vid_stride = vid_stride
        self.accuracy_level = accuracy_level

    def run(self):
        source_copy = np.copy(self.source)
        return self.inference(source_copy, self.model, s='image:')

    def inference(self, source, model, s):
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        bs = 1  # batch_size
        im0 = source  # original image
        # Preprocessing
        self.transforms = classify_transforms(imgsz[0])
        im = self.transforms(im0)

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        with dt[0]:
            im = torch.Tensor(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
            results = model(im)

        # NMS
        with dt[2]:
            pred = F.softmax(results, dim=1)

        # Process predictions
        for i, prob in enumerate(pred):  # per image
            seen += 1

            s += '%gx%g ' % im.shape[2:]  # print string
            # im0 = im0.astype('uint8')
            # annotator = Annotator(im0, example=str(names), pil=True)

            # Print results
            top5i = prob.argsort(0, descending=True)[:self.accuracy_level].tolist()  # top 5 indices
            s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in top5i)}, "

            # Write results
            results = []
            for j in top5i:
                results.append({"name": names[j], "index": j, "confidence": f"{prob[j]:.3f}"})
            # text = '\n'.join(f'{prob[j]:.3f} {names[j]}' for j in top5i)
            # annotator.text((32, 32), text, txt_color=(255, 255, 255))
            #
            # im0 = annotator.result()

        # Print time (inference-only)
        LOGGER.info(f'{s}{dt[1].dt * 1E3:.1f}ms')

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

        return results, im0
