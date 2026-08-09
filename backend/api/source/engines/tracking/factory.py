from .bytetrack import ByteTrack
from .deepsort import DeepSORT

TRACKERS = {
    "deepsort": DeepSORT,
    "bytetrack": ByteTrack,
}


def create_tracker(config):
    """Build the tracker selected by config["tracker"]["type"].

    Defaults to DeepSORT so configs without a tracker section keep the
    previous behavior.
    """
    tracker_type = (config.get("tracker") or {}).get("type", "deepsort")
    try:
        tracker_cls = TRACKERS[tracker_type]
    except KeyError:
        raise ValueError(
            f"Unknown tracker '{tracker_type}'. "
            f"Available: {sorted(TRACKERS)}")
    return tracker_cls(config)
