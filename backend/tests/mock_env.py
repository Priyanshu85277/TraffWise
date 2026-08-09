import sys
import types
from unittest.mock import MagicMock

class BaseDummyClass:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return MagicMock()

class AutoModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        val = MagicMock()
        setattr(self, name, val)
        return val

def setup_mocks():
    if "torch" in sys.modules and not isinstance(sys.modules["torch"], AutoModule):
        return

    mods = [
        "numpy", "cv2", "torch", "torch.nn", "torchvision",
        "torchvision.ops", "torchvision.transforms", "torchvision.transforms.functional",
        "torchvision.models", "torchvision.models.detection",
        "torchvision.models.detection.transform", "torchvision.models.detection.backbone_utils",
        "torchvision.models.detection.rpn", "PIL", "PIL.Image",
        "ultralytics", "ultralytics.nn", "ultralytics.nn.modules",
        "ultralytics.nn.modules.head", "ultralytics.nn.modules.transformer",
        "ultralytics.trackers", "ultralytics.trackers.byte_tracker",
        "easyocr", "cloudinary", "cloudinary.uploader", "cloudinary.api",
        "scipy", "scipy.spatial", "scipy.spatial.distance", "scipy.optimize", "yaml",
        "deep_sort_realtime", "deep_sort_realtime.deepsort_tracker", "dotenv"
    ]

    for mname in mods:
        if mname not in sys.modules:
            m = AutoModule(mname)
            m.__file__ = f"<dummy {mname}>"
            m.__path__ = []
            sys.modules[mname] = m

    mock_bytes = MagicMock()
    mock_bytes.tobytes.return_value = b"jpegdata"
    sys.modules["cv2"].imencode = MagicMock(return_value=(True, mock_bytes))

    sys.modules["torch.nn"].Module = BaseDummyClass
    sys.modules["torchvision.models.detection.transform"].GeneralizedRCNNTransform = BaseDummyClass
    sys.modules["torchvision.models.detection.rpn"].AnchorGenerator = BaseDummyClass
    sys.modules["torchvision.models.detection.backbone_utils"].BackboneWithFPN = BaseDummyClass
    sys.modules["torchvision.models.detection"].FasterRCNN = BaseDummyClass
    sys.modules["deep_sort_realtime.deepsort_tracker"].DeepSort = BaseDummyClass
