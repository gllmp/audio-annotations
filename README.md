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

- Confirm that your input file is a WAV (mono and 16 kHz recommended), otherwise convert it

```bash
ffmpeg -i input_audio.mp3 -ac 1 -ar 16000 input_audio.wav
```

- Run the script from the command line:

```bash
python audio_segmenter.py <input_audio.wav> <output.json>
```

For example:

```bash
python audio_segmenter.py audio/bells.wav output/output-bell.json
python audio_segmenter.py audio/evening.wav output/output-evening.json
```

- Check the console output. It will print time ranges with detected labels.
- View the JSON in the specified output file. Each entry includes:
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
