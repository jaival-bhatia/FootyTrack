import cv2
import torch
from trackers import Tracker
import supervision as sv

SRC_VIDEO = r"C:\Users\JaivalBhatia\Desktop\All_codes\football_project_opencv\input_videos\08fd33_4.mp4"
DST_VIDEO = r"C:\Users\JaivalBhatia\Desktop\All_codes\football_project_opencv\output_videos\tracked_output.mp4"
MODEL_PT  = r"C:\Users\JaivalBhatia\Desktop\All_codes\football_project_opencv\models\best.pt"

def draw_boxes_opencv(frame, detections, class_names):
    for i, box in enumerate(detections.xyxy):
        x1, y1, x2, y2 = map(int, box)
        track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
        cls_id = int(detections.class_id[i]) if detections.class_id is not None else -1

        label = f"{class_names.get(cls_id, 'Obj')} {track_id} ({conf:.2f})"
        color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def main():
    cap = cv2.VideoCapture(SRC_VIDEO)
    if not cap.isOpened():
        print("[main] ❌ Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[main] Loaded video: {total} frames at {fps:.2f} FPS, size: {width}x{height}")

    out = cv2.VideoWriter(DST_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracker = Tracker(model_path=MODEL_PT, device="cuda:0", half=True, imgsz=640)
    class_names = tracker.model.model.names

    frame_idx = 0
    batch_size = 1  # Process one frame at a time to reduce GPU load

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process one frame as a list of one
        detections = tracker.get_object_tracks([frame], conf=0.4)[0]

        annotated = draw_boxes_opencv(frame, detections, class_names)
        out.write(annotated)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"[main] Processed {frame_idx}/{total} frames")

        # Clear cache every frame to avoid GPU memory fragmentation
        torch.cuda.empty_cache()

    cap.release()
    out.release()
    print(f"[main] ✅ Done. Saved to {DST_VIDEO}")

if __name__ == "__main__":
    main()
