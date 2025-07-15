from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout,
                           QPushButton, QLabel, QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
import TSVReader 

class LandingWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setWindowTitle("Dependency Analysis")
        self.resize(1000, 700)
        
        screen = self.app.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
        
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        with open("src/gui/styles/GlobalStyles.css", "r") as f:
            style_sheet = f.read()
        self.setStyleSheet(style_sheet)

        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = QLabel("Dependency Analysis")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        
       
        create_button = QPushButton("Create TSV for Analysis")
        create_button.setMinimumWidth(300)
        create_button.setMinimumHeight(50)
        create_button.clicked.connect(self.choose_directory)
        create_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        
        analyze_button = QPushButton("Analyze TSV")
        analyze_button.setMinimumWidth(300)
        analyze_button.setMinimumHeight(50)
        analyze_button.clicked.connect(self.analyze_tsv)
        analyze_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        layout.addSpacing(50)
        layout.addWidget(create_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(analyze_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(50)

    def choose_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory for TSV Creation",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            print(f"Selected directory: {directory}")
            # gotta add tsvreader shit here
            

    def analyze_tsv(self):
        
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select TSV File",
            "",
            "TSV Files (*.tsv)"
        )
        if file_name:
            print(f"Selected TSV file: {file_name}")

        
        

