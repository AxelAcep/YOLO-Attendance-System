from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QFrame, QHBoxLayout, QGraphicsDropShadowEffect, QGridLayout, QSizePolicy, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QColor, QFont
import os

width = 1600
height = 900

# Path gambar relatif
def get_image_path(image_name: str) -> str:
    return os.path.join(os.path.dirname(__file__), f'../Assets/{image_name}')

class Profile(QWidget):

    def create_block(self, color: str, text: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(f"background-color: {color}; border-radius: 10px;")
        label = QLabel(text, frame)
        label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(frame)
        layout.addWidget(label)
        return frame

    def create_header(self, name:str) -> QFrame:
        header = QFrame(self)
        header.setStyleSheet("background-color: white; border: none;")
        header.setFixedHeight(int(height * 0.14))

        # Drop shadow effect
        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setOffset(0, 10)
        shadow_effect.setBlurRadius(25)
        shadow_effect.setColor(Qt.gray)
        header.setGraphicsEffect(shadow_effect)

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 40, 0)  # Biar ada padding kiri-kanan

        # Logo
        logo_path = get_image_path("Logo.png")
        logo_label = QLabel(header)
        logo_pixmap = QPixmap(logo_path)
        logo_width = int(logo_pixmap.width() * (int(height * 0.12) / logo_pixmap.height()))
        logo_pixmap = logo_pixmap.scaled(logo_width, int(height * 0.12), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        header_layout.addWidget(logo_label)

        header_layout.addStretch()

        # Greeting Text
        greeting_label = QLabel(f"{name}", header)
        greeting_label.setStyleSheet("""
            font-family: 'Nunito Sans', 'Segoe UI', Sans-serif;
            font-size: 21px;
            font-weight: bold;
            color: #6E7491;
        """)
        header_layout.addWidget(greeting_label)

        return header

    def __init__(self, main_window):
        
        super().__init__()
        self.main_window = main_window
        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: white;")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Membuat dan menambahkan header
        header = self.create_header("Profile") #Ganti dari API
        main_layout.addWidget(header)

        # Body dengan background transparan
        self.body = QFrame(self)
        self.body.setStyleSheet("background-color: transparent;")
        self.body.setFixedHeight(int(height * 0.86))

        grid = QGridLayout()
        grid.setSpacing(20)
        self.body.setLayout(grid)

        grid.addWidget(self.create_block("#32a852", ""), 0, 0, 1, 1)

        # Menambahkan body ke layout utama
        main_layout.addWidget(self.body)

        # Menetapkan layout utama
        self.setLayout(main_layout)