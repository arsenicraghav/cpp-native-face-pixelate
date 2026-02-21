CXX := c++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic

OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null)

TARGET := build/face_pixelate_cpp
SRC := src/main.cpp

.PHONY: all clean run ensure-opencv

all: ensure-opencv $(TARGET)

ensure-opencv:
	@if pkg-config --exists opencv4 || pkg-config --exists opencv; then \
		echo "OpenCV detected via pkg-config."; \
	else \
		echo "OpenCV not found. Attempting to install via Homebrew..."; \
		if command -v brew >/dev/null 2>&1; then \
			brew install opencv; \
		else \
			echo "Homebrew is not installed. Install Homebrew first, then rerun make."; \
			exit 1; \
		fi; \
	fi

$(TARGET): $(SRC)
	@mkdir -p build
	@OPENCV_CFLAGS="$$(pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)"; \
	OPENCV_LIBS="$$(pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null)"; \
	if [ -z "$$OPENCV_CFLAGS" ] || [ -z "$$OPENCV_LIBS" ]; then \
		echo "OpenCV pkg-config metadata still missing after setup."; \
		exit 1; \
	fi; \
	$(CXX) $(CXXFLAGS) $$OPENCV_CFLAGS $(SRC) -o $(TARGET) $$OPENCV_LIBS

run: $(TARGET)
	./$(TARGET) --model ./face_detection_yunet_2023mar.onnx --camera 0 --pixel-block 28 --face-padding 0.5 --hold-frames 20 --score-threshold 0.8

clean:
	rm -rf build
