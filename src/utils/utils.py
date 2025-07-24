
import os
import sys
import torch

sys.path.append('/opt/project/capsules/Yolov5/src/lib/yolov5')

from sdks.novavision.src.base.download import Download
from sdks.novavision.src.base.logger import LoggerManager
from sdks.novavision.src.base.application import Application
from capsules.Yolov5.src.lib.yolov5.utils.torch_utils import select_device
from capsules.Yolov5.src.lib.yolov5.models.common import DetectMultiBackend

logger = LoggerManager()
application = Application()

weights = {
    "PPENano.pt": "https://drive.google.com/file/d/15_zb0bLzi6RZCifcQO90DuzO8VNAesLP/view?usp=sharing",
    "PPESmall.pt": "https://drive.google.com/file/d/1kV17EuapBgDR_KEoixA49yx7S4jwh5MA/view?usp=sharing",
    "PPEMedium.pt": "https://drive.google.com/file/d/1We_Ud0cncGqUwJUNdr_KgutoDhiauzIq/view?usp=sharing",
    "PPELarge.pt": "https://drive.google.com/file/d/1EmiyJcnrsuwn2rupRtTTaWNW67NsqVsR/view?usp=sharing",
    "PPEX.pt": "https://drive.google.com/file/d/1TxjqH3vib_nYckdxtvmhDL6skrbq2KBX/view?usp=sharing"
}

def download_weights(url, weight_name):
    if not os.path.exists(f"/storage/{weight_name}"):
        if Download.download_from_drive(url, f"/storage/{weight_name}") is None:
            logger.error(f"PpeDetection - Model ({weight_name}) download failed!!")

    weight_path = f"/storage/{weight_name}"
    return weight_path

def load_models(config):
    models = {}
    weight = application.get_param(config=config, name="Weights")
    weight_path = download_weights(url=weights[weight], weight_name=weight)
    config_device = application.get_param(config=config, name="ConfigDevice")
    device = select_device('cuda:0' if config_device == "GPU" and torch.cuda.is_available() else 'cpu')

    if config_device == "GPU" and torch.cuda.is_available():
        if application.get_param(config=config, name="Half"):
            model = DetectMultiBackend(weight_path, device=device, fp16=True)
        else:
            model = DetectMultiBackend(weight_path, device=device, fp16=False)
    else:
        model = DetectMultiBackend(weight_path, device=device, fp16=False)

    models["model"] = model
    models["device"] = device
    return models
