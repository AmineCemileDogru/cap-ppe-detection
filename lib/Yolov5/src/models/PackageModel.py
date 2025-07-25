
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Detection, Package, Inputs, Configs, Outputs, Response, Request, Output, Input, Config, Image, KeyPoints


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class Detection(Detection):
    keyPoints: Optional[List[KeyPoints]] = None
    imgUID: str


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class CustomFieldStorageID(Config):
    name: Literal["Id"] = "Id"
    value: int
    type: Literal["number"] = "number"
    field: Literal["filePicker"] = "filePicker"

    class Config:
        json_schema_extra = {
            "class": "portalium\\storage\\widgets\\FilePicker",
            "options": {
                "multiple": 0,
                "returnAttribute": [
                    "name"
                ],
                "name": "app::logo_wide"
            }
        }
        title = "Storage Source"


class CustomFieldStorage(Config):
    name: Literal["storageid"] = "storageid"
    storageID: CustomFieldStorageID
    value: Literal["storageid"] = "storageid"
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Storage ID"


class ConfigCustom(Config):
    customFieldStorage: CustomFieldStorage
    name: Literal["ConfigCustom"] = "ConfigCustom"
    value: Literal["ConfigCustom"] = "ConfigCustom"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Custom"


class ConfigWeightsYolov5Segmentn(Config):
    name: Literal["Yolov5Segmentn"] = "Yolov5Segmentn"
    value: Literal["yolov5n-seg.pt"] = "yolov5n-seg.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "yolov5n-seg.pt"


class ConfigWeightsYolov5Segments(Config):
    name: Literal["Yolov5Segments"] = "Yolov5Segments"
    value: Literal["yolov5s-seg.pt"] = "yolov5s-seg.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "yolov5s-seg.pt"


class ConfigWeightsYolov5Segmentm(Config):
    name: Literal["Yolov5Segmentm"] = "Yolov5Segmentm"
    value: Literal["yolov5m-seg.pt"] = "yolov5m-seg.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "yolov5m-seg.pt"


class ConfigWeightsYolov5Segmentl(Config):
    name: Literal["Yolov5Segmentl"] = "Yolov5Segmentl"
    value: Literal["yolov5l-seg.pt"] = "yolov5l-seg.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "yolov5l-seg.pt"


class ConfigWeightsYolov5Segmentx(Config):
    name: Literal["Yolov5Segmentx"] = "Yolov5Segmentx"
    value: Literal["yolov5x-seg.pt"] = "yolov5x-seg.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "yolov5x-seg.pt"


class SegmentConfigWeights(Config):
    """
        It corresponds to weights of the models. Yolov5(n/s/m) is recommended for mobile deployments
        Yolov5(l/x) is recommended for cloud deployments.
    """
    name: Literal["Weights"] = "Weights"
    value: Union[
        ConfigWeightsYolov5Segmentn, ConfigWeightsYolov5Segments, ConfigWeightsYolov5Segmentm, ConfigWeightsYolov5Segmentl, ConfigWeightsYolov5Segmentx, ConfigCustom]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Weights"


class ConfigWeightsYolov5Classifyn(Config):
    name: Literal["Yolov5Classifyn"] = "Yolov5Classifyn"
    value: Literal["yolov5n-cls.pt"] = "yolov5n-cls.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "yolov5n-cls.pt"


class ConfigWeightsYolov5Classifys(Config):
    name: Literal["Yolov5Classifys"] = "Yolov5Classifys"
    value: Literal["yolov5s-cls.pt"] = "yolov5s-cls.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "yolov5s-cls.pt"


class ConfigWeightsYolov5Classifym(Config):
    name: Literal["Yolov5Classifym"] = "Yolov5Classifym"
    value: Literal["yolov5m-cls.pt"] = "yolov5m-cls.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "yolov5m-cls.pt"


class ConfigWeightsYolov5Classifyl(Config):
    name: Literal["Yolov5Classifyl"] = "Yolov5Classifyl"
    value: Literal["yolov5l-cls.pt"] = "yolov5l-cls.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "yolov5l-cls.pt"


class ConfigWeightsYolov5Classifyx(Config):
    name: Literal["ClassifyYolov5x"] = "ClassifyYolov5x"
    value: Literal["yolov5x-cls.pt"] = "yolov5x-cls.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "yolov5x-cls.pt"


class ClassifyConfigWeights(Config):
    """
        It corresponds to weights of the models. Yolov5(n/s/m) is recommended for mobile deployments
        Yolov5(l/x) is recommended for cloud deployments.
    """
    name: Literal["Weights"] = "Weights"
    value: Union[ConfigWeightsYolov5Classifyn, ConfigWeightsYolov5Classifys, ConfigWeightsYolov5Classifym, ConfigWeightsYolov5Classifyl, ConfigWeightsYolov5Classifyx, ConfigCustom]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Weights"


class ConfigShowAccuracyTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ConfigShowAccuracyFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigShowAccuracy(Config):
    """
        It allows display of accuracy values on the image.
    """
    name: Literal["ShowAccuracy"] = "ShowAccuracy"
    value: Union[ConfigShowAccuracyTrue, ConfigShowAccuracyFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Show Accuracy"


class ConfigTop5(Config):
    name: Literal["ConfigTop5"] = "ConfigTop5"
    value: Literal["5"] = "5"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Top-5"


class ConfigTop1(Config):
    name: Literal["ConfigTop1"] = "ConfigTop1"
    value: Literal["1"] = "1"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Top-1"


class ConfigModelAccuracy(Config):
    """
        Top-1 accuracy refers to the model's accuracy in making its top prediction, whereas Top-5 accuracy considers its top five predictions.
    """
    name: Literal["ConfigModelAccuracy"] = "ConfigModelAccuracy"
    value: Union[ConfigTop1, ConfigTop5]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Model Accuracy"


class ConfigWeightsYolov5n(Config):
    name: Literal["Yolov5n"] = "Yolov5n"
    value: Literal["yolov5n.pt"] = "yolov5n.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Yolov5n.pt"


class ConfigWeightsYolov5s(Config):
    name: Literal["Yolov5s"] = "Yolov5s"
    value: Literal["yolov5s.pt"] = "yolov5s.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Yolov5s.pt"


class ConfigWeightsYolov5m(Config):
    name: Literal["Yolov5m"] = "Yolov5m"
    value: Literal["yolov5m.pt"] = "yolov5m.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Yolov5m.pt"


class ConfigWeightsYolov5l(Config):
    name: Literal["Yolov5l"] = "Yolov5l"
    value: Literal["yolov5l.pt"] = "yolov5l.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Yolov5l.pt"


class ConfigWeightsYolov5x(Config):
    name: Literal["Yolov5x"] = "Yolov5x"
    value: Literal["yolov5x.pt"] = "yolov5x.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Yolov5x.pt"


class DetectConfigWeights(Config):
    """
        It corresponds to weights of the models. Yolov5(n/s/m) is recommended for mobile deployments
        Yolov5(l/x) is recommended for cloud deployments.
    """
    name: Literal["Weights"] = "Weights"
    value: Union[ConfigWeightsYolov5n, ConfigWeightsYolov5s, ConfigWeightsYolov5m, ConfigWeightsYolov5l, ConfigWeightsYolov5x, ConfigCustom]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Weights"


class ConfigConfidentThreshold(Config):
    """
        Detected Objects with a confidence score below this threshold will be ignored or filtered out.
    """
    name: Literal["ConfidentThreshold"] = "ConfidentThreshold"
    value: float = Field(default=0.3, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Confidence Threshold"


class ConfigIOUThreshold(Config):
    """
        It is necessary for Non-Maximum Suppression (NMS) which is used to filter redundant bounding boxes for the same object. A higher IOU threshold will result in fewer bounding boxes after NMS.
    """
    name: Literal["IOUThreshold"] = "IOUThreshold"
    value: float = Field(default=0.3, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "IOU Threshold"


class ConfigHalfTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ConfigHalfFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigHalf(Config):
    """
        It enables half-precision (FP16) inference, which can speed up model inference.
    """
    name: Literal["Half"] = "Half"
    value: Union[ConfigHalfTrue, ConfigHalfFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Half"


class ConfigDeviceGPU(Config):
    name: Literal["ConfigDeviceGPU"] = "ConfigDeviceGPU"
    configHalf: ConfigHalf
    value: Literal["GPU"] = "GPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU"


class ConfigDeviceCPU(Config):
    name: Literal["ConfigDeviceCPU"] = "ConfigDeviceCPU"
    value: Literal["CPU"] = "CPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CPU"


class ConfigDevice(Config):
    """
        It refers to whether the model should run on a CPU or a GPU.
        You can select the device type for inference or training process.
    """
    name: Literal["ConfigDevice"] = "ConfigDevice"
    value: Union[ConfigDeviceCPU, ConfigDeviceGPU]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Device"


class DetectInputs(Inputs):
    inputImage: InputImage


class DetectConfigs(Configs):
    configWeights: DetectConfigWeights
    configDevice: ConfigDevice
    configConfidentThreshold: ConfigConfidentThreshold
    configIOUThreshold: ConfigIOUThreshold


class DetectOutputs(Outputs):
    outputDetections: OutputDetections


class DetectRequest(Request):
    inputs: Optional[DetectInputs]
    configs: DetectConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class DetectResponse(Response):
    outputs: DetectOutputs


class ClassifyInputs(Inputs):
    inputImage: InputImage


class ClassifyConfigs(Configs):
    configWeights: ClassifyConfigWeights
    configDevice: ConfigDevice
    configModelAccuracy: ConfigModelAccuracy
    configShowAccuracy: ConfigShowAccuracy


class ClassifyOutputs(Outputs):
    outputDetections: OutputDetections


class ClassifyRequest(Request):
    inputs: Optional[ClassifyInputs]
    configs: ClassifyConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ClassifyResponse(Response):
    outputs: ClassifyOutputs


class SegmentInputs(Inputs):
    inputImage: InputImage


class SegmentConfigs(Configs):
    configWeights: SegmentConfigWeights
    configDevice: ConfigDevice
    configConfidentThreshold: ConfigConfidentThreshold
    configIOUThreshold: ConfigIOUThreshold


class SegmentOutputs(Outputs):
    outputDetections: OutputDetections


class SegmentRequest(Request):
    inputs: Optional[SegmentInputs]
    configs: SegmentConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class SegmentResponse(Response):
    outputs: SegmentOutputs


class DetectExecutor(Config):
    name: Literal["Detect"] = "Detect"
    value: Union[DetectRequest, DetectResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ClassifyExecutor(Config):
    name: Literal["Classify"] = "Classify"
    value: Union[ClassifyRequest, ClassifyResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Classification"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class SegmentExecutor(Config):
    name: Literal["Segment"] = "Segment"
    value: Union[SegmentRequest, SegmentResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Segmentation"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    """
        Three fundamental tasks can be selected to analyze the content of the image.
    """
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[DetectExecutor, ClassifyExecutor, SegmentExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["Yolov5"] = "Yolov5"