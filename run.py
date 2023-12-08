import sys
from PyQt5.QtWidgets import QApplication
from Core.main import MyMainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MyMainWindow()
    ui.show()
    sys.exit(app.exec_())
