from typing import Literal, Optional
from pydantic import BaseModel


class ModelRequest(BaseModel):
    model_type: Literal["yolo11", "yolo26", "rtdetrv2", "faster_rcnn"]


class CameraRequest(BaseModel):
    camera_id: str


class CaptureRequest(BaseModel):
    camera_id: str
    model_type: Literal["yolo11", "yolo26", "rtdetrv2", "faster_rcnn"]


class Violation(BaseModel):
    id: str
    plate: str
    type: str
    status: str
    date: str
    location: str
    evidence: str
    speed: Optional[str] = None
    signalTime: Optional[str] = None
    laneDetails: Optional[str] = None


class AnnotationToggleRequest(BaseModel):
    show_annotations: bool


class SystemParameters(BaseModel):
    cameraId: str = "1"
    speedEstimationEnabled: bool = True
    speedLimit: float = 60
    overspeedBuffer: float = 5
    maxHistorySeconds: float = 3.0

    redLightDetectionEnabled: bool = True
    maxTrackRLV: int = 50

    wrongLaneDetectionEnabled: bool = True
    angleThreshold: float = 90
    straightThreshold: float = 30
    dotThreshold: float = -0.5
    toleranceTime: float = 3

    confidenceThreshold: float = 0.5
    iouThreshold: float = 0.45
    maxAge: int = 15
