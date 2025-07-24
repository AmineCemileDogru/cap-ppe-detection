
from sdks.novavision.src.helper.package import PackageHelper
from capsules.PpeDetection.src.models.PackageModel import PackageConfigs, ConfigExecutor, PPEExecutor, PackageModel, PPEResponse, PPEOutputs, OutputDetections


def build_response(context):
    outputDetections = OutputDetections(value=context.detection)
    ppeOutputs = PPEOutputs(outputDetections=outputDetections)
    pperesp = PPEResponse(outputs=ppeOutputs)
    ppeExecutor = PPEExecutor(value=pperesp)
    executor = ConfigExecutor(value=ppeExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel