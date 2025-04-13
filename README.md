# Audio Annotations

This script analyzes an audio file by splitting it into segments, classifying the dominant sound in each segment using the [YAMNet](https://tfhub.dev/google/yamnet/1) model from TensorFlow Hub, and merging consecutive segments that share the same label. The final results are stored in a JSON file with time ranges and labels.

## Requirements

- **Python 3.10+** (3.10 or 3.11 recommended)
- **FFmpeg**
- **Python packages**:
  - `tensorflow`
  - `tensorflow-hub`
  - `librosa`
  - `numpy`

You can install these with:
```bash
pip install tensorflow tensorflow-hub librosa numpy
````

## Usage

1. Convert your input file(s) to WAV if necessary (mono, 16 kHz recommended):

```bash
ffmpeg -i input_audio.mp3 -ac 1 -ar 16000 input_audio.wav
```

2. Run the script from the command line. You can pass:

- A single `.wav` file:

```bash
python audio_segmenter.py audio/evening.wav
```

This will output output/evening.json.

- A folder containing `.wav` files:

```bash
python audio_segmenter.py audio
```

This will process all `.wav` files in `audio/`, creating JSON files in the `output/ folder (e.g., `output/evening.json`, `output/bells.json`, etc.).

3. Check the console output. It will print time ranges with detected labels.
4. View the JSON in the specified output file. Each entry includes:
  - start (seconds)
  - end (seconds)
  - label (YAMNet class)
  - confidence (float)

Example JSON snippet:

```json
[
  {
    "start": 0.0,
    "end": 5.0,
    "label": "Speech",
    "confidence": 0.78
  },
  {
    "start": 5.0,
    "end": 10.0,
    "label": "Bird",
    "confidence": 0.62
  }
]
```

## License

This project is licensed under the [MIT License](LICENSE).
