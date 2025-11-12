from faster_whisper import WhisperModel


def main(audio_path: str, model_size: str = "small"):
    print(f"Loading model '{model_size}'...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"Transcribing: {audio_path}")
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"\nDetected language: {info.language} (prob={info.language_probability:.2f})\n")
    print("=== TRANSCRIPT ===\n")
    full_text = []
    for seg in segments:
        line = f"[{seg.start:6.2f}s -> {seg.end:6.2f}s] {seg.text}"
        print(line)
        full_text.append(seg.text)

    print("\n=== FULL TEXT ONLY ===\n")
    print(" ".join(full_text))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to your audio file")
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model size (tiny, base, small, medium, large)",
    )
    args = parser.parse_args()

    main(args.audio, args.model)
