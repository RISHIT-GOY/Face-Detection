# Face Detection and Recognition

A Python-based face detection and recognition project using OpenCV and computer vision techniques.

## Features

- **Face Detection**: Detects faces in images using Haar Cascade classifiers
- **Face Recognition**: Recognizes and identifies known faces (Me, Sister, Mother)
- **Dataset Support**: Organized dataset structure for training data

## Project Structure

```
Face Detection/
├── main.py                 # Main application script
├── dataset/               # Training dataset directory
│   ├── me/               # Images of me
│   ├── sister/           # Images of sister
│   └── mother/           # Images of mother
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy

## Installation

1. **Clone the repository** (if applicable):
```bash
git clone <repository-url>
cd Face\ Detection
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Linux/Mac
```

3. **Install dependencies**:
```bash
pip install opencv-python numpy
```

## Setup

1. **Prepare your dataset**:
   - Create folders inside the `dataset/` directory named: `me`, `sister`, `mother`
   - Add face images of each person to their respective folders
   - Supported formats: `.png`, `.jpg`, `.jpeg`

2. **Run the application**:
```bash
python main.py
```

## How It Works

1. The application scans the dataset folder for images organized by person
2. It detects faces in each image using Haar Cascade classifiers
3. Detected faces are resized to a standard size (200x200)
4. Faces and their labels are prepared for training/recognition

## Usage

### Training Data Preparation
The `prepare_training_data()` function:
- Loads images from the dataset directory
- Detects faces using Haar Cascade
- Filters out images without detected faces
- Resizes faces to a standard size
- Returns faces and corresponding labels

## Future Enhancements

- [ ] Implement face recognition model training
- [ ] Add real-time face detection from webcam
- [ ] Support for multiple face detection in a single image
- [ ] Performance optimization and GPU support
- [ ] Web interface for predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Feel free to fork this project and submit pull requests with improvements.

## Author

Created for personal face detection and recognition project.
