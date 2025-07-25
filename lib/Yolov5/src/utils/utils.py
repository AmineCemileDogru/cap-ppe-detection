
import os
import sys
import torch
import urllib
import hashlib

sys.path.append('/opt/project/capsules/Yolov5/src/lib/yolov5')

from sdks.novavision.src.base.download import Download
from sdks.novavision.src.base.logger import LoggerManager
from sdks.novavision.src.helper.package import PackageHelper
from sdks.novavision.src.base.application import Application
from capsules.Yolov5.src.configs.config import weights_url, CONFIG
from capsules.Yolov5.src.lib.yolov5.utils.torch_utils import select_device
from capsules.Yolov5.src.lib.yolov5.models.common import DetectMultiBackend

logger = LoggerManager()


class ModelLoader:
    def __init__(self, config: dict):
        self.config = config
        self.application = Application()
        self.executor = self.application.get_param(config=config, name="ConfigExecutor")["name"]
        self.models = {}

    def download_weights(self):
        if not os.path.exists(f"/storage/{self.weight_name}"):
            if Download.download_from_drive(weights_url[self.weight_name], f"/storage/{self.weight_name}") is None:
                logger.error(f"Yolov5 - Model ({self.weight_name}) download failed!!")

        weight_path = f"/storage/{self.weight_name}"
        return weight_path

    def load_model(self):
        weight_type = self.application.get_param(config=self.config, name=CONFIG["Weights"])
        if weight_type == CONFIG["ConfigCustom"]:
            storage_id = self.application.get_param(config=self.config, name="Id")
            weight_path = load_storage(storage_id)
        else:
            self.weight_name = self.application.get_param(config=self.config, name=CONFIG["Weights"])
            weight_path = self.download_weights()

        config_device = self.application.get_param(config=self.config, name="ConfigDevice")
        device = select_device('cuda:0' if config_device == "GPU" and torch.cuda.is_available() else 'cpu')

        if config_device == "GPU" and torch.cuda.is_available():
            if self.application.get_param(config=self.config, name="Half"):
                model = DetectMultiBackend(weight_path, device=device, fp16=True)
            else:
                model = DetectMultiBackend(weight_path, device=device, fp16=False)
        else:
            model = DetectMultiBackend(weight_path, device=device, fp16=False)

        self.models["model"] = model
        self.models["device"] = device
        return self.models

def load_storage(storageID):
    result = PackageHelper.get_storage_details(storageID)
    data = result["data"]
    url_path = result["data_url"]
    name = data["name"]
    hash_file = data["hash_file"]
    file_path = f"/storage/{name}"
    storage = os.listdir("/storage")
    if name in storage:
        md5_hash_file = md5_hash(file_path)
        if md5_hash_file != hash_file:
            urllib.request.urlretrieve(url_path, file_path)
    else:
        urllib.request.urlretrieve(url_path, file_path)
    weight_path = f"/storage/{name}"
    return weight_path

def md5_hash(file_path, chunk_size=8192):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as file:
        while True:
            data = file.read(chunk_size)
            if not data:
                break
            md5_hash.update(data)
    return md5_hash.hexdigest()
