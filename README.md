# Object Tracker

A simple real-time tracker that detects hands and faces using your webcam. When nothing is detected, it tries to identify other objects in the frame.

This project uses MediaPipe for hand and face detection, which gives really accurate results even in different lighting conditions. The hand tracking shows all 21 landmarks on each hand, making it useful for gesture recognition or hand movement analysis. Face detection works quickly and can spot faces at various angles.

When you're not showing your hands or face to the camera, the program switches to object recognition mode using a neural network trained on thousands of everyday items. It'll tell you what it sees with confidence percentages.

## What it does

- Tracks up to 2 hands with landmarks and bounding boxes
- Detects faces and draws boxes around them
- Identifies everyday objects when hands/faces aren't visible
- Shows smooth tracking with color-coded boxes (cyan and magenta for hands, yellow for faces)
- Uses exponential moving average to reduce jittery movements
- Displays top predictions with percentages for object classification

## How to use

Install the required packages:
```bash
pip install opencv-python mediapipe torch torchvision pillow
```

Run the tracker:
```bash
python object_tracker.py --source 0 --display
```

Press ESC to quit.

## Options

- `--source 0` - Use webcam (0 is default camera)
- `--display` - Show the video window
- `--max_hands 2` - How many hands to track (default is 2)
- `--topk 3` - Show top 3 object predictions
- `--smooth 0.3` - Smoothing for bounding boxes (lower = smoother)

That's it! Just point your camera and see what it detects.
