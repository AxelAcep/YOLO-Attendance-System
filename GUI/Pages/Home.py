from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QFrame, QHBoxLayout, QGraphicsDropShadowEffect, QGridLayout, QSizePolicy, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QColor, QFont
import os

width = 1600
height = 900

# Path gambar relatif
def get_image_path(image_name: str) -> str:
    return os.path.join(os.path.dirname(__file__), f'../Assets/{image_name}')

class Home(QWidget):

    def CreateKelas(self):
        print("Buat Kelas button clicked")
        self.main_window.stacked_widget.setCurrentWidget(self.main_window.page7)
        #QTimer.singleShot(0, self.switch_to_dashboard)
    
    def RekamKelas(self):
        print("Rekam Kelas button clicked")
        self.main_window.stacked_widget.setCurrentWidget(self.main_window.page4)

    def ProfileKelas(self):
        print("Profile button clicked")
        self.main_window.stacked_widget.setCurrentWidget(self.main_window.page5)
        #QTimer.singleShot(0, self.switch_to_dashboard)
        
    def HistoryKelas(self):
        print("History button clicked")
        self.main_window.stacked_widget.setCurrentWidget(self.main_window.page6)
        #QTimer.singleShot(0, self.switch_to_dashboard)

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
            font-weight: bold;
            color: #018298;
            padding-right: 10px;
        """)
        kelas_label.setStyleSheet("""
            font-family: 'Nunito Sans', 'Segoe UI', Sans-serif;
            font-size: 21px;
            font-weight: semi-bold;
            color: #7C8DB0;
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

    
    def create_card(self, image_path: str, title: str, desc: str, button_action) -> QFrame:
        return CardWidget(image_path, title, desc, button_action)

    def create_custom_button(self, text: str, on_click_callback) -> QFrame:
        button = QPushButton(text, self)  # pastikan parent = self
        button.setStyleSheet("""
            QPushButton {
                background-color: #ff1938;
                color: white;
                border-radius: 8px;
                padding: 5px 10px;
                font-family: 'Nunito Sans', 'Segoe UI', 'Arial', sans-serif;
                font-size: 24px;
            }
            QPushButton:hover {
                background-color: #a30217;
            }
            QPushButton:pressed {
                background-color: #a30217;
            }
        """)
        button.clicked.connect(on_click_callback)
        button.setFixedHeight(int(height * 0.07))
        button.setFixedWidth(int(width * 0.1))

        container = QFrame(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(button)
        layout.addStretch()

        return container

    def __init__(self, main_window):

        def create_image_block(color: str, image_path: str, height_percent: float = 0.5, vertical_align: str = "center", marginleft: float = 0.0) -> QFrame:
            frame = QFrame()
            frame.setStyleSheet(f"background-color: {color}; border-radius: 10px;")

            image_label = QLabel()
            pixmap = QPixmap(image_path)

            align_map = {
                "top": Qt.AlignTop | Qt.AlignHCenter,
                "center": Qt.AlignCenter,
                "bottom": Qt.AlignBottom | Qt.AlignHCenter
            }
            image_label.setAlignment(align_map.get(vertical_align, Qt.AlignCenter))

            layout = QVBoxLayout(frame)
            layout.setContentsMargins(int(height*marginleft), 0, 0, 0)
            layout.addWidget(image_label)

            def resizeEvent(event):
                container_height = frame.height()
                target_height = int(container_height * height_percent)
                scaled = pixmap.scaledToHeight(target_height, Qt.SmoothTransformation)
                image_label.setPixmap(scaled)

            frame.resizeEvent = resizeEvent

            return frame
        
        super().__init__()
        self.main_window = main_window
        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: white;")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Membuat dan menambahkan header
        header = self.create_header("Axel Sebayang") #Ganti dari API
        main_layout.addWidget(header)

        # Body dengan background transparan
        self.body = QFrame(self)
        self.body.setStyleSheet("background-color: transparent;")
        self.body.setFixedHeight(int(height * 0.86))

        grid = QGridLayout()
        grid.setSpacing(10)
        self.body.setLayout(grid)

        grid.addWidget(create_image_block("#00FFFFFF", "./Assets/home-headertext.png", height_percent=0.55, vertical_align="center"), 0, 0, 2, 4)
        grid.addWidget(self.create_card("./Assets/home-edit.png", "Buat/Edit Kelas", "Kelola atau Membuat Kelas dan Individu", self.CreateKelas), 2, 0, 6, 1)
        grid.addWidget(self.create_card("./Assets/home-capture.png", "Mulai/Rekam", "Rekam dan Pantau Kehadiran Individu", self.RekamKelas), 2, 1, 6, 1)
        grid.addWidget(self.create_card("./Assets/home-profile.png", "Profile", "Kelola Identitas Anda", self.ProfileKelas), 2, 2, 6, 1)
        grid.addWidget(self.create_card("./Assets/home-history.png", "History", "Recap Hasil Kehadiran Individu", self.HistoryKelas), 2, 3, 6, 1)
        grid.addWidget(self.create_block("#00FFFFFF", ""), 8, 0, 1, 4)
        #grid.addWidget(create_image_block("#00FFFFFF", "./Assets/UPN.png", height_percent=0.5, vertical_align="center"), 8, 0, 1, 4)
        grid.addWidget(self.create_custom_button("Logout", lambda: self.main_window.stacked_widget.setCurrentWidget(self.main_window.page1)), 9, 0, 1, 4)




        # Menambahkan body ke layout utama
        main_layout.addWidget(self.body)

        # Menetapkan layout utama
        self.setLayout(main_layout)

class CardWidget(QFrame):
    def __init__(self, image_path: str, title: str, desc: str, button_action):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # Styling
        self.setStyleSheet("""
            QFrame {
                background-color: #e8e8e8;
                border: none;
                border-radius: 12px;
            }
            QLabel {
                font-family: 'Nunito Sans', 'Segoe UI', Sans-serif;
                color: #6E7491;
            }
            QPushButton {
                font-family: 'Nunito Sans', 'Segoe UI', Sans-serif;
                color: white;
                background-color: #007AFF;
                border: none;
                border-radius: 6px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #473afc;
            }
        """)

        # Drop shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 60))  # translucent black
        self.setGraphicsEffect(shadow)

        self.image_path = image_path
        self.pixmap = QPixmap(image_path)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # Layout utama
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(self.image_label)

        # Layout bawah
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(5)

        text_layout = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setFont(QFont("Nunito Sans", 11, QFont.Bold))

        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setFont(QFont("Nunito Sans", 10))

        text_layout.addWidget(title_label)
        text_layout.addWidget(desc_label)

        button = QPushButton("Aksi")
        button.clicked.connect(button_action)
        button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        button.setFixedWidth(80)
        button.setFixedHeight(30)

        bottom_layout.addLayout(text_layout)
        bottom_layout.addWidget(button)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        card_height = self.height()
        image_height = int((7 / 10) * card_height)
        if not self.pixmap.isNull():
            scaled = self.pixmap.scaledToHeight(image_height, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)
