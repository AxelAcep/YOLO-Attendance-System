from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import QTimer, Qt
import os

width = 1600
height = 900

# Path gambar dan GIF
image_path = os.path.join(os.path.dirname(__file__), '../Assets/OnBoarding.png')
loading_gif_path = os.path.join(os.path.dirname(__file__), '../Assets/Loading.gif')  # Path GIF animasi loading

class OnBoarding(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.main_window = main_window

        self.setFixedSize(width, height)  # Ukuran tetap jendela

        # Membuat QLabel untuk menampilkan gambar latar belakang
        self.background = QLabel(self)
        pixmap = QPixmap(image_path)
        if pixmap.isNull(): 
            print("Gambar tidak ditemukan!")
        else:
            print("Gambar berhasil dimuat.")

        self.background.setPixmap(pixmap)  # Menetapkan gambar latar belakang pada QLabel
        self.background.setScaledContents(True)  # Agar gambar menyesuaikan dengan ukuran widget
        self.background.setGeometry(0, 0, width, height)  # Mengatur posisi dan ukuran background
        
        # Membuat QLabel untuk animasi loading (GIF)
        self.loading_label = QLabel(self)
        self.loading_movie = QMovie(loading_gif_path)

        if not self.loading_movie.isValid():
            print("GIF loading tidak ditemukan!")
        else:
            print("GIF loading berhasil dimuat.")
        
        self.loading_label.setMovie(self.loading_movie)  # Menghubungkan GIF dengan QLabel
        self.loading_label.setAlignment(Qt.AlignCenter)  # Menempatkan animasi di tengah widget
        self.loading_label.setGeometry(int(width/1.6), int(height/1.4), int(width/24), int(width/24))  # Posisi GIF di tengah bawah
        self.loading_label.setScaledContents(True)

        # Mulai animasi loading
        self.loading_movie.start()

        # **Kontrol manual posisi widget** tanpa menggunakan layout untuk z-order yang tepat
        self.background.raise_()  # Pastikan background tetap di belakang
        self.loading_label.raise_()  # GIF muncul di depan background

        # Mengatur timer untuk berpindah ke halaman dashboard setelah 5 detik
        QTimer.singleShot(5000, self.switch_to_dashboard)

    def switch_to_dashboard(self):
        # Mengakses halaman dashboard dari MainWindow dan beralih ke halaman tersebut
        self.main_window.stacked_widget.setCurrentWidget(self.main_window.page1)
