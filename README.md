# C++ Native Face Pixelate (macOS)

A beginner-friendly macOS app that uses OpenCV's YuNet neural network model to detect faces from your webcam and pixelate them in real time.

## What this project does

- Opens your webcam feed.
- Detects faces using **YuNet** (a lightweight neural network model in OpenCV).
- Applies pixelation over detected faces for privacy.
- Keeps pixelation active for a few frames even if detection briefly drops.

## Project files

- `src/main.cpp`: Main C++ application logic.
- `face_detection_yunet_2023mar.onnx`: YuNet model file used by OpenCV.
- `Makefile`: Build, run, and clean commands.
- `.gitignore`: Ignores build output.

## Prerequisites

1. macOS
2. Xcode Command Line Tools (`clang`, `make`)
3. OpenCV C++ libraries

The `Makefile` checks for OpenCV using `pkg-config`.
If OpenCV is missing and Homebrew is available, it automatically runs `brew install opencv`.

## Quick start (first-time users)

```bash
cd /Users/raghvendra/CodexProjects/cpp-native-face-pixelate
make
make run
```

When the camera window opens:
- Press `q` or `Esc` to quit.

## Camera permissions on macOS

If no camera window appears or frames are black:

1. Open **System Settings**.
2. Go to **Privacy & Security** -> **Camera**.
3. Enable camera access for your terminal app (for example, Terminal or iTerm).
4. Close and relaunch the app.

## Build and run manually

### Build

```bash
make
```

### Run

```bash
./build/face_pixelate_cpp \
  --model ./face_detection_yunet_2023mar.onnx \
  --camera 0 \
  --pixel-block 28 \
  --face-padding 0.5 \
  --hold-frames 20 \
  --score-threshold 0.8
```

## Command options

- `--model <path>`: Path to YuNet `.onnx` model.
- `--camera <index>`: Camera index (`0` is default webcam).
- `--pixel-block <int>`: Pixelation strength. Larger value = chunkier pixels.
- `--face-padding <float>`: Expands face box before pixelating.
- `--hold-frames <int>`: Reuses last detected face boxes when detector flickers.
- `--score-threshold <float>`: Confidence threshold for detection.
- `--nms-threshold <float>`: Overlap filtering threshold.
- `--top-k <int>`: Max candidate boxes before overlap filtering.

## Good defaults for beginners

Use these values first:

- `--pixel-block 28`
- `--face-padding 0.5`
- `--hold-frames 20`
- `--score-threshold 0.8`

If your face is sometimes exposed:

- Increase `--face-padding` to `0.6` or `0.7`
- Increase `--hold-frames` to `25` or `30`
- Lower `--score-threshold` to `0.7`

## Troubleshooting

### `OpenCV pkg-config metadata still missing`

- Ensure OpenCV is installed.
- Ensure `pkg-config` can find OpenCV.
- Retry `make`.

### Camera does not open

- Verify camera permission for terminal app in macOS settings.
- Ensure no other app is exclusively locking the camera.

### Model file error

- Confirm `face_detection_yunet_2023mar.onnx` exists in project root.
- Or pass the correct path using `--model`.

## Clean build output

```bash
make clean
```

## Command Reference

See `COMMANDS.md` for the sanitized list of relevant setup/build/run/publish commands.
