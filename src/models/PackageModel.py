
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Detection, Package, Inputs, Configs, Outputs, Response, Request, Output, Input, Config,Image


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image],Image]
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
    imgUID: str


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class ConfigConfidentThreshold(Config):
    """
    (0.0-1.0) Represents the overlap threshold value.
    """
    name: Literal["ConfidentThreshold"] = "ConfidentThreshold"
    value: float = Field(default=0.3, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Confidence Threshold"


class ConfigIOUThreshold(Config):
    """
    (0.0-1.0) Represents the overlap threshold value.
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


class ConfigPPEWeights1(Config):
    name: Literal["PPENano.pt"] = "PPENano.pt"
    value: Literal["PPENano.pt"] = "PPENano.pt"
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Personel Protective Equipment Model - Nano"


class ConfigPPEWeights2(Config):
    name: Literal["PPESmall.pt"] = "PPESmall.pt"
    value: Literal["PPESmall.pt"] = "PPESmall.pt"
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Personel Protective Equipment Model - Small"


class ConfigPPEWeights3(Config):
    name: Literal["PPEMedium.pt"] = "PPEMedium.pt"
    value: Literal["PPEMedium.pt"] = "PPEMedium.pt"
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Personel Protective Equipment Model - Medium"


class ConfigPPEWeights4(Config):
    name: Literal["PPELarge.pt"] = "PPELarge.pt"
    value: Literal["PPELarge.pt"] = "PPELarge.pt"
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Personel Protective Equipment Model - Large"


class ConfigPPEWeights5(Config):
    name: Literal["PPEX.pt"] = "PPEX.pt"
    value: Literal["PPEX.pt"] = "PPEX.pt"
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Personel Protective Equipment Model - Extra Large"


class PPEConfigWeights(Config):
    """
    It corresponds to weights of the models.
    """
    name: Literal["Weights"] = "Weights"
    value: Union[ConfigPPEWeights1, ConfigPPEWeights2,ConfigPPEWeights3,ConfigPPEWeights4,ConfigPPEWeights5]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Weights"


class PPEInputs(Inputs):
    inputImage: InputImage


class PPEConfigs(Configs):
    configWeights: PPEConfigWeights
    configDevice: ConfigDevice
    configConfidentThreshold: ConfigConfidentThreshold
    configIOUThreshold: ConfigIOUThreshold


class PPEOutputs(Outputs):
    outputDetections: OutputDetections


class PPERequest(Request):
    inputs: Optional[PPEInputs]
    configs: PPEConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class PPEResponse(Response):
    outputs: PPEOutputs


class PPEExecutor(Config):
    name: Literal["PpeDetection"] = "PpeDetection"
    value: Union[PPERequest, PPEResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Personel Protective Equipment"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[PPEExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["PPE"] = "PPE"
