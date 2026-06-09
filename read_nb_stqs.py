import sys
import json

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    try:
        from stqs_questions import QUESTIONS

        print(len(QUESTIONS))
        return
    except ImportError:
        pass

    with open("main.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "stqs_questions" in src:
            print("cell", i)
            print(src)


if __name__ == "__main__":
    main()
