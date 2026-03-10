from pathlib import Path


if __name__ == "__main__":
    print(Path(__file__).with_suffix(".pto").read_text(encoding="utf-8"))
