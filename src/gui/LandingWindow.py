from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout,
                           QPushButton, QLabel, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor, QFontDatabase
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

        fonts_path = Path(__file__).parent / 'fonts' / 'corm' / 'cormorant-garamond'
        for font_file in fonts_path.glob('*.ttf'):
            QFontDatabase.addApplicationFont(str(font_file))
    
        
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        with open("src/gui/styles/GlobalStyles.css", "r") as f:
            style_sheet = f.read()
        self.setStyleSheet(style_sheet)

        self.layout = QVBoxLayout(central_widget)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = QLabel("Dependency Analysis")
        title.setObjectName("title")
        self.layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        
       
        create_button = QPushButton("Create TSV for Analysis")
        create_button.setMinimumWidth(300)
        create_button.setMinimumHeight(50)
        create_button.clicked.connect(self.choose_directory)
        create_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        create_button.setObjectName("regButton")

        
        analyse_button = QPushButton("Analyse TSV")
        analyse_button.setMinimumWidth(300)
        analyse_button.setMinimumHeight(50)
        analyse_button.clicked.connect(self.analyse_tsv)
        analyse_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        analyse_button.setObjectName("regButton")
        
        self.layout.addSpacing(50)
        self.layout.addWidget(create_button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(analyse_button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addSpacing(50)

    def choose_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory for TSV Creation",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            project_root = Path(__file__).resolve().parent.parent.parent
            rel_path = Path(directory).relative_to(project_root)
            print(f"Selected directory: {rel_path}")
            self.progress_label = QLabel("Starting...")
            self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.layout.addWidget(self.progress_label)
            
            try:
                # Process files and update progress
                for success, message in TSVReader.process_directory(
                    directory=str(rel_path),
                    omit_javalang=False,  # Add checkboxes in GUI for these options
                    omit_javaall=False,
                    output='aggregated_output.tsv'
                ):
                    if not success:
                        QMessageBox.critical(self, "Error", message)
                        break
                    self.progress_label.setText(message)
                    QApplication.processEvents()  # Keep GUI responsive
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
            finally:
                # Clean up progress display
                self.progress_label.deleteLater()
            

    def analyse_tsv(self):
        
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select TSV File",
            "",
            "TSV Files (*.tsv)"
        )
        if file_name:
            print(f"Selected TSV file: {file_name}")

        
        

