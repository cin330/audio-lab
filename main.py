# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import AudioLabWindow


def main():
    app = QApplication(sys.argv)
    window = AudioLabWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
