#!/usr/bin/env python3
import os
import sys
import json

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

def load_labels(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def classify_segments(audio_path, model, labels, segment_length=5.0, overlap=1.0, sr=16000):
    samples, _ = librosa.load(audio_path, sr=sr, mono=True)
    seg_samples = int(segment_length * sr)
    hop_samples = int((segment_length - overlap) * sr)
    start = 0
    segments = []

    while start < len(samples):
        end = min(start + seg_samples, len(samples))
        segment_data = samples[start:end]
        tensor_data = tf.convert_to_tensor(segment_data, dtype=tf.float32)
        scores, _, _ = model(tensor_data)
        avg_scores = tf.reduce_mean(scores, axis=0).numpy()
        top_idx = np.argmax(avg_scores)

        segments.append({
            "start": round(start / sr, 2),
            "end": round(end / sr, 2),
            "label": labels[top_idx],
            "confidence": float(avg_scores[top_idx])
        })

        if end == len(samples):
            break
        start += hop_samples

    return segments

def merge_labels(segments):
    if not segments:
        return segments
    merged = []
    current = segments[0].copy()

    for seg in segments[1:]:
        if seg["label"] == current["label"]:
            current["end"] = seg["end"]
            current["confidence"] = max(current["confidence"], seg["confidence"])
        else:
            merged.append(current)
            current = seg.copy()
    merged.append(current)
    return merged

def analyze_audio(audio_path, model, labels, output_json):
    raw_segments = classify_segments(audio_path, model, labels)
    merged_segments = merge_labels(raw_segments)
    results = []

    for seg in merged_segments:
        results.append({
            "start": seg["start"],
            "end": seg["end"],
            "label": seg["label"],
            "confidence": round(seg["confidence"], 3)
        })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("Analysis complete. Results written to:", output_json)
    for seg in merged_segments:
        print(f'{seg["start"]:.2f}-{seg["end"]:.2f} {seg["label"]}')

def main(audio_file, output_file):
    model_url = "https://tfhub.dev/google/yamnet/1"
    model = hub.load(model_url)
    label_path = os.path.join(os.path.dirname(__file__), "labels", "yamnet_label_list.txt")
    labels = load_labels(label_path)
    analyze_audio(audio_file, model, labels, output_file)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python audio_segmenter.py <input_audio.wav> <output.json>")
        sys.exit(1)

    audio_file = sys.argv[1]
    output_file = sys.argv[2]
    main(audio_file, output_file)
