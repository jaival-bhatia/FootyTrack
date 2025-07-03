from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path, device="cuda:0", half=True, imgsz=640, byte_params=None):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.model.fuse()
        self.model.model.half() if half else self.model.model.float()
        self.device = device
        self.imgsz = imgsz

        # Remove unsupported args
        self.tracker = sv.ByteTrack()  # 'byte_params' not supported in your version

    def get_object_tracks(self, frames, conf=0.25):
        results = self.model.predict(
            frames, conf=conf, device=self.device, imgsz=self.imgsz, verbose=False
        )

        tracked_detections = []
        for r in results:
            det = sv.Detections.from_ultralytics(r)
            tracked = self.tracker.update_with_detections(det)
            tracked_detections.append(tracked)
        return tracked_detections
