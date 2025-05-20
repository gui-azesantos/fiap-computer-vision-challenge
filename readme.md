# Video Emotion Analysis with DeepFace

This project provides a Python script to analyze emotions in a video using the **DeepFace** library, generate an annotated output video with bounding boxes and emotion labels, and save detailed emotion data to an Excel file.

---

## Features

- **Emotion Detection:** Utilizes DeepFace to detect and analyze emotions in each frame of a video.
- **Real-time Annotation:** Draws bounding boxes around detected faces and labels them with the dominant emotion and its probability in the output video.
- **Data Export:** Saves comprehensive emotion data (frame number, bounding box coordinates, dominant emotion, and all emotion probabilities) to an Excel spreadsheet for further analysis.
- **Progress Tracking:** Uses `tqdm` to display a progress bar during video analysis.

---

## Prerequisites

Before you begin, ensure you have **Python 3.10** installed on your system.

---

## Setup

Follow these steps to set up the project in a virtual environment:

### 1. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies. Open your terminal or command prompt and run:

```bash
python3.10 -m venv deepface_env
```

2. Activate the Virtual Environment
   On Windows:

```bash
.\venv\Scripts\activate

```

```bash
source venv/bin/activate
```

3. Install Dependencies
   With your virtual environment activated, install the required libraries:

```bash
pip install opencv-python deepface pandas tqdm tf-keras
```

### Usage

Put the input video file you want to analyze in the same directory as the main.py script and rename it to input.mp4.

Execute the script from your activated virtual environment:

```bash
python main.py
```

### Expected Output

Upon successful execution, the script will generate two files in the same directory:

- **`output_analysis.mp4`**  
  An annotated video displaying detected faces with bounding boxes and corresponding emotion labels.

- **`video_analysis.xlsx`**  
  An Excel file containing detailed information about emotions detected in each frame, including:
  - `frame`: frame number
  - `x, y, width, height`: coordinates and dimensions of the detected face's bounding box
  - `dominant_emotion`: the most prominent emotion detected
  - Probabilities for each emotion: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
