import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QWidget, QMessageBox, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage, QAction
from PyQt6.QtCore import Qt


class MainWindow(QMainWindow):
    """
    Main application window for LifDetector.
    Provides GUI for loading AVI files, displaying video frames, running flash detection, and generating PDF reports.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LifDetector")
        # Set initial window size to 75% of the current display
        screen = self.screen() if hasattr(self, 'screen') else None
        if screen is None:
            screen = QApplication.primaryScreen()
        geometry = screen.geometry()
        width = int(geometry.width() * 0.75)
        height = int(geometry.height() * 0.75)
        self.resize(width, height)

        # --- Top panel ---
        from PyQt6.QtWidgets import QLabel as QtLabel, QLineEdit, QSlider
        from PyQt6.QtCore import Qt as QtCoreQt
        self.open_button = QPushButton("Select AVI File")
        self.open_button.setFixedWidth(120)
        self.open_button.clicked.connect(self.open_file)
        self.avi_label = QtLabel("AVI File:")
        self.avi_path_box = QLineEdit()
        self.avi_path_box.setReadOnly(True)
        self.avi_path_box.setMinimumWidth(300)
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.open_button)
        top_layout.addSpacing(10)
        top_layout.addWidget(self.avi_label)
        top_layout.addWidget(self.avi_path_box, stretch=1)
        top_widget = QWidget()
        top_widget.setLayout(top_layout)

        # --- Left panel ---
        self.detect_button = QPushButton("Detect Flashes")
        self.detect_button.clicked.connect(self.detect_flashes)
        self.report_button = QPushButton("Create PDF Report")
        self.report_button.clicked.connect(self.create_pdf_report)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.detect_button)
        left_layout.addWidget(self.report_button)
        left_layout.addStretch(1)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # --- Main panel (frame display) ---
        self.image_label = QLabel("Open an AVI file to begin.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # --- Frame slider and frame number display ---
        self.frame_slider = QSlider(QtCoreQt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.slider_frame_changed)
        self.frame_number_box = QLineEdit()
        self.frame_number_box.setReadOnly(True)
        self.frame_number_box.setFixedWidth(240)
        self.frame_number_box.setFixedHeight(40)
        self.frame_number_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_number_box.setStyleSheet("background-color: #2ecc40; color: white; font-weight: bold; font-size: 20px; border-radius: 8px;")

        # --- Video controls ---
        from PyQt6.QtWidgets import QToolButton
        from PyQt6.QtGui import QIcon, QFont
        button_size = 64  # pixels (default is usually 32)
        font_size = 32   # Large font for button icons
        button_font = QFont()
        button_font.setPointSize(font_size)

        self.play_button = QToolButton()
        self.play_button.setText('▶')
        self.play_button.setFixedSize(button_size, button_size)
        self.play_button.setFont(button_font)
        self.play_button.clicked.connect(self.play_video)

        self.stop_button = QToolButton()
        self.stop_button.setText('■')
        self.stop_button.setFixedSize(button_size, button_size)
        self.stop_button.setFont(button_font)
        self.stop_button.clicked.connect(self.stop_video)

        self.ff_button = QToolButton()
        self.ff_button.setText('⏩')
        self.ff_button.setFixedSize(button_size, button_size)
        self.ff_button.setFont(button_font)
        self.ff_button.clicked.connect(self.fast_forward)

        self.rev_button = QToolButton()
        self.rev_button.setText('⏪')
        self.rev_button.setFixedSize(button_size, button_size)
        self.rev_button.setFont(button_font)
        self.rev_button.clicked.connect(self.reverse)
        controls_layout = QHBoxLayout()
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.rev_button)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.ff_button)
        controls_layout.addStretch(1)

        # --- Layout composition ---
        main_layout = QVBoxLayout()
        main_layout.addWidget(top_widget)
        content_layout = QHBoxLayout()
        content_layout.addWidget(left_widget)
        # Main panel with image and controls
        video_panel_layout = QVBoxLayout()
        video_panel_layout.addWidget(self.image_label, stretch=1)
        video_panel_layout.addWidget(self.frame_slider)
        video_panel_layout.addWidget(self.frame_number_box)
        video_panel_layout.addLayout(controls_layout)
        content_layout.addLayout(video_panel_layout, stretch=1)
        main_layout.addLayout(content_layout)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def play_video(self):
        if not self.cap:
            return
        import threading
        import time
        self._playing = True
        def run():
            while self._playing and self.cap:
                pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.show_frame(pos)
                if not self._playing:
                    break
                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                    break
                time.sleep(1.0 / max(1, self.cap.get(cv2.CAP_PROP_FPS)))
        threading.Thread(target=run, daemon=True).start()


    def stop_video(self):
        self._playing = False


    def fast_forward(self):
        if not self.cap:
            return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.show_frame(pos + 10)


    def reverse(self):
        if not self.cap:
            return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.show_frame(max(0, pos - 10))

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open AVI...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def create_pdf_report(self):
        QMessageBox.information(self, "PDF Report", "PDF report generation not implemented yet.")


    def detect_flashes(self):
        if not self.video_path:
            QMessageBox.warning(self, "No Video", "Please open an AVI file first.")
            return
        import threading
        from .detection import detect_flashes

        def run_detection():
            flashes = detect_flashes(self.video_path, progress_callback=self.update_frame_number_box)
            if flashes:
                msg = f"Flashes detected at frames: {flashes}"
            else:
                msg = "No flashes detected."
            def show_result():
                QMessageBox.information(self, "Flash Detection", msg)
            self.run_on_main_thread(show_result)

        threading.Thread(target=run_detection, daemon=True).start()

    def update_frame_number_box(self, frame_idx):
        from PyQt6.QtCore import QMetaObject, Qt as QtCoreQt
        def update():
            self.frame_number_box.setText(f"Frame: {frame_idx}")
        QMetaObject.invokeMethod(self, update, QtCoreQt.ConnectionType.QueuedConnection)

    def run_on_main_thread(self, func):
        from PyQt6.QtCore import QMetaObject, Qt as QtCoreQt
        QMetaObject.invokeMethod(self, func, QtCoreQt.ConnectionType.QueuedConnection)


    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open AVI File", "", "AVI Files (*.avi)"
        )
        if file_path:
            self.video_path = file_path
            self.avi_path_box.setText(file_path)
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open video file.")
                return
            # Enable and set up the slider
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_slider.setEnabled(True)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(max(0, total_frames - 1))
            self.frame_slider.setValue(0)
            self.show_frame(0)


    def show_frame(self, frame_idx):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "Error", "Could not read frame.")
            return
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        # Update slider and frame number box
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_idx)
        self.frame_slider.blockSignals(False)
        self.frame_number_box.setText(f"Frame: {frame_idx}")

    def slider_frame_changed(self, value):
        self.show_frame(value)
