"""NodeGraphQt visual pipeline editor entry point."""
import sys

from PySide6.QtWidgets import QApplication

from sigflow_editor.window import EditorWindow


def main():
    app = QApplication(sys.argv)
    window = EditorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
