from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QFrame, QHBoxLayout, QGraphicsDropShadowEffect, QGridLayout, QPushButton, QLineEdit
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap
import os

width = 1600
height = 900

class Dashboard(QWidget):
    def button_pressed(self):
        if self.modal.isVisible():
            self.modal.hide()
            self.overlay.hide()
            self.body_darkened = False
        else:
            self.overlay.setGeometry(self.rect())
            self.overlay.show()

            self.modal.resize(int(width * 0.3), int(height * 0.4))
            self.modal.move(
                (self.width() - self.modal.width()) // 2,
                (self.height() - self.modal.height()) // 2
            )
            self.modal.show()
            self.body_darkened = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.overlay.isVisible():
            self.overlay.setGeometry(self.rect())
        if self.modal.isVisible():
            self.modal.move(
                (self.width() - self.modal.width()) // 2,
                (self.height() - self.modal.height()) // 2
            )
    
    def loginPressed(self):
        print("Login button clicked")
        QTimer.singleShot(0, self.switch_to_dashboard)

    def switch_to_dashboard(self):
        # Mengakses halaman dashboard dari MainWindow dan beralih ke halaman tersebut
        self.main_window.stacked_widget.setCurrentWidget(self.main_window.page2)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
    
        def create_label(text, font_size=16, bold=False, color="#6E7491") -> QLabel:
            label = QLabel(text)
            weight = "bold" if bold else "normal"
            label.setStyleSheet(f"""
                color: {color};
                font-size: {font_size}px;
                font-family: 'Nunito Sans', 'Segoe UI', sans-serif;
                font-weight: {weight};
            """)
            return label

        def create_input(placeholder: str, password: bool = False) -> QLineEdit:
            input_field = QLineEdit()
            input_field.setPlaceholderText(placeholder)
            input_field.setEchoMode(QLineEdit.Password if password else QLineEdit.Normal)
            input_field.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 16px;
                    font-family: 'Nunito Sans', 'Segoe UI', sans-serif;
                }
            """)
            return input_field

        def create_button(text: str, color: str = "#605DEC", callback=None) -> QPushButton:
            button = QPushButton(text)
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 18px;
                    font-family: 'Nunito Sans', 'Segoe UI', sans-serif;
                }}
                QPushButton:hover {{
                    background-color: #514CD4;
                }}
            """)
            if callback:
                button.clicked.connect(callback)
            return button

        # Helper untuk membuat blok gambar
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

        # Helper untuk membuat tombol
        def create_custom_button(text: str, on_click_callback) -> QPushButton:
            button = QPushButton(text)
            button.setStyleSheet("""
                QPushButton {
                    background-color: #007AFF;
                    color: white;
                    border-radius: 8px;
                    padding: 5px 10px;
                    font-family: 'Nunito Sans', 'Segoe UI', sans-serif;
                    font-size: 24px;
                    min-width: 150px;
                    min-height: 50px;
                }
                QPushButton:hover {
                    background-color: #005FCC;
                }
                QPushButton:pressed {
                    background-color: #0047A0;
                }
            """)
            button.clicked.connect(on_click_callback)
            return button

        # Layout utama
        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: white;")

        # Header
        header = QFrame(self)
        header.setStyleSheet("background-color: white; border: none;")
        header.setFixedHeight(int(height * 0.14))

        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setOffset(0, 10)
        shadow_effect.setBlurRadius(25)
        shadow_effect.setColor(Qt.gray)
        header.setGraphicsEffect(shadow_effect)

        header_layout = QHBoxLayout(header)
        logo_path = os.path.join(os.path.dirname(__file__), '../Assets/Logo.png')
        logo_label = QLabel(header)
        logo_pixmap = QPixmap(logo_path)
        logo_width = int(logo_pixmap.width() * (int(height * 0.12) / logo_pixmap.height()))
        logo_pixmap = logo_pixmap.scaled(logo_width, int(height * 0.12), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        header_layout.addWidget(logo_label)
        header_layout.addStretch()

        # Body
        self.body = QFrame(self)
        self.body.setStyleSheet("background-color: transparent;")
        self.body.setFixedHeight(int(height * 0.86))
        self.body_darkened = False

        grid = QGridLayout()
        grid.setSpacing(10)
        self.body.setLayout(grid)

        grid.addWidget(create_image_block("#00FFFFFF", "./Assets/db-text1.png", height_percent=0.65, vertical_align="bottom", marginleft=0.1), 0, 0, 6, 1, alignment=Qt.AlignLeft)
        grid.addWidget(create_image_block("#00FFFFFF", "./Assets/db-text2.png", height_percent=0.76, vertical_align="center", marginleft=0.1), 6, 0, 1, 1, alignment=Qt.AlignLeft)
        grid.addWidget(create_custom_button("Masuk", self.button_pressed), 7, 0, 2, 1, alignment=Qt.AlignCenter)
        grid.addWidget(create_image_block("#00FFFFFF", "./Assets/db-img1.png", height_percent=0.7), 0, 1, 8, 1)
        grid.addWidget(create_image_block("#00FFFFFF", "./Assets/UPN.png", height_percent=0.4, vertical_align="top"), 8, 1, 2, 1)

        # Layout utama
        layout = QVBoxLayout(self)
        layout.addWidget(header)
        layout.addWidget(self.body)
        layout.setContentsMargins(0, 0, 0, 10)
        layout.setSpacing(0)
        self.setLayout(layout)

        # Overlay (di atas semua)
        # Overlay dan Modal
        self.overlay = QWidget(self)
        self.overlay.setStyleSheet("background-color: rgba(0, 0, 0, 120);")
        self.overlay.hide()

        self.modal = QFrame(self)
        self.modal.setStyleSheet("background-color: white; border-radius: 12px;")
        self.modal.hide()

        modal_layout = QVBoxLayout(self.modal)
        modal_layout.setContentsMargins(30, 30, 30, 30)
        modal_layout.setSpacing(15)

        # Header: Judul + Tombol X
        header_layout = QHBoxLayout()
        title_label = create_label("Masuk untuk Mira", font_size=24, bold=True)
        close_btn = QPushButton("X")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                font-size: 18px;
                font-weight: bold;
                color: #6E7491;
                border: none;
            }
            QPushButton:hover {
                color: red;
            }
        """)
        close_btn.clicked.connect(self.button_pressed)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(close_btn)

        # Subjudul dan Form Input
        subtitle = create_label("Masuk menggunakan alamat email dan sandi", font_size=16)

        self.email_input = create_input("Alamat Email")
        self.password_input = create_input("Kata Sandi", password=True)

        login_button = create_button("Masuk", "#605DEC", lambda: self.loginPressed())

        # Tambah ke layout modal
        modal_layout.addLayout(header_layout)
        modal_layout.addWidget(subtitle)
        modal_layout.addWidget(self.email_input)
        modal_layout.addWidget(self.password_input)
        modal_layout.addWidget(login_button)
        modal_layout.addStretch()

