import subprocess
import sys
from pathlib import Path


def main() -> None:
    import config.settings  # noqa: F401

    root = Path(__file__).resolve().parent
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(root / "app.py")],
            cwd=str(root),
        )
    )


if __name__ == "__main__":
    main()
