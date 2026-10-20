from flask import Flask, render_template, request, jsonify
import wave
import os
import tempfile
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


def analyze_audio(filepath):
    with wave.open(filepath, "rb") as audio:
        rate = audio.getframerate()
        width = audio.getsampwidth()
        channels = audio.getnchannels()
        frames = audio.readframes(audio.getnframes())

    if width != 2:
        raise ValueError("Please upload a 16-bit PCM WAV file.")

    signal = np.frombuffer(frames, dtype="<i2").astype(np.float32)

    if channels > 1:
        signal = signal[:len(signal) - len(signal) % channels]
        signal = signal.reshape(-1, channels).mean(axis=1)

    if len(signal) < rate // 2:
        raise ValueError("Audio must be at least 0.5 seconds long.")

    if len(signal) > rate * 90:
        raise ValueError("Audio must be 90 seconds or shorter.")

    signal /= 32768.0
    rms = float(np.sqrt(np.mean(signal ** 2)))

    return {
        "duration_seconds": round(len(signal) / rate, 2),
        "sample_rate": rate,
        "rms_level": round(rms, 4),
        "result": "Audio features extracted",
        "warning": (
            "This prototype does not determine whether a voice "
            "is real or AI-generated."
        )
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded = request.files.get("audio")

    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Please select a WAV file."}), 400

    if not uploaded.filename.lower().endswith(".wav"):
        return jsonify({"error": "Only WAV files are supported."}), 400

    path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp:
            path = temp.name
            uploaded.save(path)

        return jsonify(analyze_audio(path))

    except (ValueError, wave.Error) as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Audio analysis failed")
        return jsonify({"error": "Could not process audio."}), 500

    finally:
        if path and os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    app.run(debug=True)
