# Simple Face Recognition with Gender and Age Detection

This project was made based on real time face recognition project https://github.com/medsriha/real-time-face-recognition.git and gender & age detection project https://github.com/smahesh29/Gender-and-Age-Detection

## Installation
Make sure you have python 3.07 or more.

```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python
- opencv-contrib-python
- pillow
- pyyaml
- argparse

## Configuration

All settings are stored in `src/config.yaml`:
- Camera settings (resolution, device index)
- Face detection parameters
- Training parameters
- File paths
- Confidence threshold (how confident the model has to be to recognize a face)

### 1. Capture Face Data
Run `DataCollection.py` to capture training images:
```bash
python src/DataCollection.py
```
- Enter your name when prompted
- :rotating_light: The script captures 120 images of your face. Make sure to have a good lighting and move your head around to capture different angles.
- Keep your face centered in the frame
- Images are saved in the `images` folder
- Your name and ID are stored in `names.json`
- Press 'ESC' to exit early

Format of `names.json`:
```json
{
    "1": "Jon",
    "2": "Janss"
}
```

### 2. Train the Model
Run `FaceTrainer.py` to create the recognition model:
```bash
python src/FaceTrainer.py
```
- Processes all images in the `images` folder
- Creates a trained model file `trainer.yml`
- Shows number of faces trained
Note: Training images are saved as: `Users-{id}-{number}.jpg`

### 3. Run Face Recognition
Run `FaceRecogGenderAgeDetect.py` to start real-time recognition:
```bash
python src/FaceRecogGenderAgeDetect.py
```
- Your webcam will open and start recording
- Recognizes faces in real-time
- Shows name and confidence level and gender & age prediction
- Press 'ESC' to exit




