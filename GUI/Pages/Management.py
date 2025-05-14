from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QFrame, QHBoxLayout, QGraphicsDropShadowEffect, QGridLayout, QSizePolicy, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QColor, QFont
import os

width = 1600
height = 900

# Path gambar relatif
def get_image_path(image_name: str) -> str:
    return os.path.join(os.path.dirname(__file__), f'../Assets/{image_name}')

class Management(QWidget):

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

        profile_label = QLabel()
        profile_label.setFixedSize(int(height* 0.05), int(height* 0.05))

        header_layout.addStretch()

        # Greeting Text
        dashboard_label = QLabel(f"Dashboard", header)
        kelas_label = QLabel(f"Kelas", header)
        record_label = QLabel(f"Record", header)
        history_label = QLabel(f"History", header)

        dashboard_label.setStyleSheet("""
            font-family: 'Nunito Sans', 'Segoe UI', Sans-serif;
            font-size: 21px;
            font-weight: semi-bold;
            color: #7C8DB0;
            padding-right: 10px;
        """)
        kelas_label.setStyleSheet("""
            font-family: 'Nunito Sans', 'Segoe UI', Sans-serif;
            font-size: 21px;
            font-weight: bold;
            color: #018298;
            padding-right: 10px;
        """)
        record_label.setStyleSheet("""
            font-family: 'Nunito Sans', 'Segoe UI', Sans-serif;
            font-size: 21px;
            font-weight: semi-bold;
            color: #7C8DB0;
            padding-right: 10px;
        """)
        history_label.setStyleSheet("""
            font-family: 'Nunito Sans', 'Segoe UI', Sans-serif;
            font-size: 21px;
            font-weight: semi-bold;
            color: #7C8DB0;
            padding-right: 10px;
        """)

        profile_label.setStyleSheet("""
            background-color: #d9d9d9; /* Warna abu-abu muda sebagai placeholder */
            border-radius: 20px; /* 40/2 supaya benar-benar bulat */
        """)

        header_layout.addWidget(dashboard_label)
        header_layout.addWidget(kelas_label)
        header_layout.addWidget(record_label)
        header_layout.addWidget(history_label)

        header_layout.addWidget(profile_label)
        
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
        header = self.create_header("Management Students")
        main_layout.addWidget(header)

        # Body
        self.body = QFrame(self)
        self.body.setStyleSheet("background-color: transparent;")
        self.body.setFixedHeight(int(height * 0.86))
        self.body.setFixedWidth(int(width * 0.93))

        # Layout body jadi VBox (vertical)
        body_layout = QVBoxLayout(self.body)
        body_layout.setSpacing(10)  # jarak antar bagian

        # Part 1: 10%
        part1 = self.create_block("#32a852", "Part 1")
        part1.setFixedHeight(int(height * 0.08))
        body_layout.addWidget(part1)

        # Part 2: 15%
        part2 = self.create_block("#3278a8", "Part 2")
        part2.setFixedHeight(int(height * 0.1))
        body_layout.addWidget(part2)

        # Part 3: 15%
        part3 = self.create_block("#a83297", "Part 3")
        part3.setFixedHeight(int(height * 0.1))
        body_layout.addWidget(part3)

        # Part 4: 60%
        part4 = self.create_block("#a89f32", "Part 4")
        part4.setFixedHeight(int(height * 0.5))
        body_layout.addWidget(part4)

        # ❗ Bungkus body di tengah
        center_layout = QHBoxLayout()
        center_layout.addStretch()
        center_layout.addWidget(self.body)
        center_layout.addStretch()

        main_layout.addLayout(center_layout)

        self.setLayout(main_layout)
