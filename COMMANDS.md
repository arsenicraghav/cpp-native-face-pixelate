# Command Log (C++ Native Face Pixelate)

This list contains only commands that were relevant to building, running, and publishing this app.
All personally identifiable details (usernames, machine names, absolute local paths, tokens) are removed.

## Environment setup

```bash
brew install opencv
```

Purpose: Installs OpenCV C++ libraries used by the app.

## Build commands

```bash
make
make clean && make
```

Purpose: Compiles `src/main.cpp` into `build/face_pixelate_cpp`.

## Run commands

```bash
make run
./build/face_pixelate_cpp \
  --model ./face_detection_yunet_2023mar.onnx \
  --camera 0 \
  --pixel-block 28 \
  --face-padding 0.5 \
  --hold-frames 20 \
  --score-threshold 0.8
```

Purpose: Launches webcam-based face detection and pixelation.

Note: Camera launch depends on macOS camera permissions for the terminal app.

## Git commands used for this project

```bash
git init
git add -A
git commit -m "Initial commit: C++ macOS YuNet face pixelation app"
git commit -m "Improve beginner docs and code comments; switch to Makefile build flow"
```

Purpose: Initializes and records project history.

## GitHub publish commands

```bash
gh repo create <github-user>/<repo-name> --public --source . --remote origin --push
git push -u origin main
```

Purpose: Creates the remote repository and publishes the project.

## Verification commands

```bash
git status --short
git log --oneline --decorate --max-count=10
git remote -v
```

Purpose: Confirms clean working tree, commit history, and remote configuration.
