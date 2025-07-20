# ⚽ FootyTrack -Lightweight, powerful tracker for players, goalkeepers, and referees in football.
<img width="1919" height="1015" alt="Screenshot 2025-07-21 051406" src="https://github.com/user-attachments/assets/503c710f-250f-4a4d-b167-0cb450f4e90e" />

Football Player Tracking using YOLOv5, OpenCV, and ByteTrack

This project enables automatic football player detection and tracking using a custom-trained YOLOv5 model and `ByteTrack`, integrated with OpenCV. It processes an input match video, annotates each frame with player detections and track IDs using **ellipses**, and saves the output as a fully processed video.

## 🚀 Features

- 📦 Real-time object detection using YOLOv5
- 🔁 Multi-object tracking using ByteTrack
- 🎥 Frame-by-frame annotation and visualization using OpenCV
- ✅ Colored **ellipses** for easy player classification:
  - 🟢 Players: Green
  - ⚫ Referee: Black
  - ⚪ Goalkeeper: White
- 💾 Output saved as a playable MP4 file

## 🗂️ Project Structure

```
football_project_opencv/
├── input_videos/
│   └── 08fd33_4.mp4               # Raw input video file
├── output_videos/
│   └── tracked_output.mp4        # Annotated output video
├── models/
│   └── best.pt                   # Trained YOLOv5 model (custom)
├── main.py                       # Main tracking pipeline
├── tracker.py                    # Wrapper around YOLOv5 + ByteTrack
├── video_utils.py                # Video reading/writing helpers
└── README.md                     # Project documentation
```

## 🧰 Requirements

Install the required dependencies:

```bash
pip install opencv-python torch torchvision torchaudio ultralytics supervision numpy
```

## ▶️ How to Run

Update the paths in `main.py`:

```python
SRC_VIDEO = r"path_to_input_video.mp4"
DST_VIDEO = r"path_to_output_video.mp4"
MODEL_PT  = r"path_to_your_model/best.pt"
```

Then run:

```bash
python main.py
```

## 📈 Future Plans

- Add ball detection and trajectory tracking
- Compute player stats (e.g. distance run, heatmaps)
- Switch to YOLOv8 or MediaPipe for better performance
- Add web dashboard for viewing metrics and video replays

## 👤 Author

**Jaival Bhatia**  
🔗 [GitHub](https://github.com/jaival-bhatia)  
📧 [Email](mailto:jaivalbhatia01@gmail.com)

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for more info.
