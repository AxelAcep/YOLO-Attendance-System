# main.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from Pages.OnBoarding import OnBoarding  
from Pages.Dashboard import Dashboard 
from Pages.Home import Home  

width = 1600
height = 900

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setFixedSize(width, height)

        self.stacked_widget = QStackedWidget(self)

        self.page0 = OnBoarding(self) 
        self.page1 = Dashboard(self)
        self.page2 = Home(self)

        self.stacked_widget.addWidget(self.page0)
        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)

        self.setCentralWidget(self.stacked_widget)

        self.stacked_widget.setCurrentWidget(self.page0)


if __name__ == "__main__":
    print("✅ App Reloaded")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
