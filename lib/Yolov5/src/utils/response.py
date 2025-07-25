
from sdks.novavision.src.helper.package import PackageHelper
from capsules.Yolov5.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, OutputDetections, DetectOutputs, ClassifyOutputs, DetectExecutor, SegmentExecutor, SegmentOutputs, ClassifyExecutor, DetectResponse, SegmentResponse, ClassifyResponse, Detection


def build_response_detect(context):
    outputDetections = OutputDetections(value=context.detections)
    detectOutputs = DetectOutputs(outputDetections=outputDetections)
    detectResponse = DetectResponse(outputs=detectOutputs)
    detectExecutor = DetectExecutor(value=detectResponse)
    executor = ConfigExecutor(value=detectExecutor)
    packageConfigs = PackageConfigs(executor=executor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_segment(context):
    outputDetections = OutputDetections(value=context.detections)
    segmentOutputs = SegmentOutputs(outputDetections=outputDetections)
    segmentResponse = SegmentResponse(outputs=segmentOutputs)
    segmentExecutor = SegmentExecutor(value=segmentResponse)
    executor = ConfigExecutor(value=segmentExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_classify(context):
    outputDetections = OutputDetections(value=context.detections)
    classifyOutputs = ClassifyOutputs(outputDetections=outputDetections)
    classifyResponse = ClassifyResponse(outputs=classifyOutputs)
    classifyExecutor = ClassifyExecutor(value=classifyResponse)
    executor = ConfigExecutor(value=classifyExecutor)
    packageConfigs = PackageConfigs(executor=executor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel


