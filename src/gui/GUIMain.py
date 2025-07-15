import sys
from PyQt6.QtWidgets import QApplication
from gui.LandingWindow import LandingWindow

def main():
    app = QApplication(sys.argv)
    window = LandingWindow(app)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()