import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def save_video(frames, output_path, fps=30):
    if not frames:
        raise ValueError("No frames to save.")

    height, width = frames[0].shape[:2]

    # Set codec based on file extension
    ext = os.path.splitext(output_path)[-1].lower()
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or 'avc1' or 'H264' if supported
    else:
        raise ValueError(f"Unsupported video format: {ext}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"[video_utils] ✅ Video saved to: {output_path}")
