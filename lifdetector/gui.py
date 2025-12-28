import cv2
import numpy as np

# QT6 imports
from PyQt6.QtCore import Qt as QtCoreQt
from PyQt6.QtWidgets import QToolButton, QApplication
from PyQt6.QtWidgets import (
    QMainWindow,
    QLabel,
    QLineEdit,
    QTabWidget,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QWidget,
    QMessageBox,
    QSlider,
    QSizePolicy
)
from PyQt6.QtGui import (
    QPixmap,
    QImage,
    QAction,
    QFont
)

from .detection import AnomalyDetector

class MainWindow(QMainWindow):
    """
    Main application window for LifDetector.
    Provides GUI for loading AVI files, displaying video frames, running flash detection, and generating PDF reports.
    """

    def __init__(self):

        super().__init__()

        self.setWindowTitle("LIF Detector")

        # Set initial window size to 75% of the current display
        screen = self.screen() if hasattr(self, 'screen') else None
        if screen is None:
            screen = QApplication.primaryScreen()
        width = 960
        height = 720
        self.resize(width, height)

        # Create a tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
    
        # Create the first (detection) tab
        self.detection_tab = QWidget()
        self.tabs.addTab(self.detection_tab, "Detection")
        self.setup_detection_ui()

        # Create the second (review) tab
        self.review_tab = QWidget()
        self.tabs.addTab(self.review_tab, "Review")
        self.setup_review_ui()

        # Create the third (configuration) tab
        self.configuration_tab = QWidget()
        self.tabs.addTab(self.configuration_tab, "Configuration")
        self.setup_configuration_ui()

        # --- Layout composition ---
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs, stretch=1)

        # Create central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self._create_menu()

        # Initialize video capture object
        self.cap = None

    def setup_detection_ui(self):

        self.avi_select_button = QPushButton("Select AVI File")
        self.avi_select_button.setFixedWidth(150)
        self.avi_select_button.clicked.connect(self.open_file)
        self.avi_select_button.setStyleSheet(
            """
            background-color: "green";
            color: white;
            border-radius: 0px;
            padding: 12px 24px 12px 24px;
            """
        )

        self.avi_label = QLabel("AVI File:")
        self.avi_path_box = QLineEdit()
        self.avi_path_box.setReadOnly(True)
        self.avi_path_box.setMinimumWidth(240)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.avi_select_button)
        top_layout.addSpacing(10)
        top_layout.addWidget(self.avi_label)
        top_layout.addWidget(self.avi_path_box, stretch=1)
        top_widget = QWidget()
        top_widget.setLayout(top_layout)

        # Image display label
        self.image_label = QLabel("No video loaded")
        self.image_label.setAlignment(QtCoreQt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; color: white;")
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(32, 32)

        # Frame slider
        self.frame_slider = QSlider(QtCoreQt.Orientation.Horizontal)
        self.frame_slider.setStyleSheet(
            """
            QSlider::handle:horizontal {
                background: #5c5c5c;
                border: 1px solid #5c5c5c;
                width: 16px;
                margin: -8px 0;
                border-radius: 3px;
            }
            QSlider::groove:horizontal {
                background: #d3d3d3;
                height: 8px;
                border-radius: 0px;
            }
            """
        )
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.slider_frame_changed)

        # Frame number box
        self.frame_number_box = QLineEdit("Frame: 0")
        self.frame_number_box.setReadOnly(True)
        self.frame_number_box.setFixedWidth(100)

        # Detection count box
        self.detection_count_box = QLineEdit("Detections: 0")
        self.detection_count_box.setReadOnly(True)
        self.detection_count_box.setFixedWidth(120)

        # Playback control buttons
        button_size = 40
        button_font = QFont()
        button_font.setPointSize(16)
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

        self.plus1_button = QToolButton()
        # Use a standard 'step forward' icon
        from PyQt6.QtGui import QIcon
        from PyQt6.QtWidgets import QStyle
        style = QApplication.style()
        step_forward_icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward)
        self.plus1_button.setIcon(step_forward_icon)
        self.plus1_button.setIconSize(self.plus1_button.sizeHint())
        self.plus1_button.setFixedSize(button_size, button_size)
        self.plus1_button.setFont(button_font)
        self.plus1_button.clicked.connect(self.step_forward)

        self.minus1_button = QToolButton()
        self.minus1_button.setText('<<')
        self.minus1_button.setFixedSize(button_size, button_size)
        self.minus1_button.setFont(button_font)
        self.minus1_button.clicked.connect(self.step_backward)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.frame_number_box)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.minus1_button)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.plus1_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.detection_count_box)

        # Detect button: full width, orange
        self.detect_button = QPushButton("Detect Flashes")
        self.detect_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.detect_button.setMinimumHeight(40)
        self.detect_button.clicked.connect(self.run_detection)
        self.detect_button.setStyleSheet(
            """
            QPushButton {
                background-color: orange;
                color: white;
                border-radius: 0px;
                padding: 12px 0px 12px 0px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #ffd699;
                color: #eeeeee;
            }
            """
        )

        self.quit_button = QPushButton("Quit")
        self.quit_button.setFixedWidth(150)
        self.quit_button.clicked.connect(self.close)
        self.quit_button.setStyleSheet(
            """
            background-color: "darkred";
            color: white;
            border-radius: 0px;
            padding: 12px 24px 12px 24px;
            """
        )

        # Create an hbox for the detect and quit buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.detect_button)
        button_layout.addWidget(self.quit_button)

        # Detection tab subpanel layout
        detection_panel_layout = QVBoxLayout()
        detection_panel_layout.addWidget(top_widget)
        detection_panel_layout.addWidget(self.image_label, stretch=1)
        detection_panel_layout.addWidget(self.frame_slider)
        detection_panel_layout.addLayout(controls_layout)
        detection_panel_layout.addLayout(button_layout)

        self.detection_tab.setLayout(detection_panel_layout)

    def setup_review_ui(self):

        review_layout = QVBoxLayout()
        review_label = QLabel("Detection Results will be shown here.")
        review_label.setAlignment(QtCoreQt.AlignmentFlag.AlignCenter)
        review_layout.addWidget(review_label)
        
        self.report_button = QPushButton("Create Report")
        self.report_button.setFixedWidth(150)
        self.report_button.clicked.connect(self.create_pdf_report)
        self.report_button.setStyleSheet(
           """
            background-color: "darkred";
            color: white;
            border-radius: 0px;
            padding: 12px 24px 12px 24px;
            """
        )
        review_layout.addWidget(self.report_button)
        self.review_tab.setLayout(review_layout)

    def setup_configuration_ui(self):

        config_layout = QVBoxLayout()

        param_grid = QGridLayout()

        param_setup = {
            "MAD Threshold (typically between 3 and 7)": 5.0,
            "Min Area (pix) (typically between 5 and 10)": 5,
            "Max Area (pix) (typically less than 100)": 75,
            "Min Duration (s)": 0.05,
            "Max Duration (s)": 0.15
        }

        # Store references to labels and boxes for later retrieval
        self.param_labels = []
        self.param_boxes = []
        
        for i, name in enumerate(param_setup.keys()):
            label = QLabel(name)
            label.setAlignment(QtCoreQt.AlignmentFlag.AlignLeft | QtCoreQt.AlignmentFlag.AlignVCenter)
            box = QLineEdit()
            box.setFixedWidth(120)
            box.setText(str(param_setup[name]))
            self.param_labels.append(label)
            self.param_boxes.append(box)
            param_grid.addWidget(label, i, 0)
            param_grid.addWidget(box, i, 1)

        # Add the parameter grid to the Configuration tab (Tab 3)
        config_tab_layout = QVBoxLayout()
        config_tab_layout.addLayout(param_grid)
        config_tab_layout.addStretch(1)
        self.configuration_tab.setLayout(config_tab_layout)


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

    def step_forward(self):
        if not self.cap:
            return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.show_frame(pos + 1)

    def step_backward(self):
        if not self.cap:
            return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.show_frame(max(0, pos - 1))

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

    def run_detection(self):

        if not hasattr(self, 'video_path') or not self.video_path:
            QMessageBox.warning(self, "No Video", "Please select an AVI file first.")
            return
        
        # Gray out the detection button and change text
        self.detect_button.setEnabled(False)
        self.detect_button.setText("Detecting...")

        flash_list = []

        # Hardwire block size to 32 frames for now
        block_size = 32

        for frame_idx in range(0, 320, block_size):

            self.show_frame(frame_idx)
            self.frame_number_box.setText(f"Frame: {frame_idx}")
            QApplication.processEvents()

            detector = AnomalyDetector(
                video_path=self.video_path,
                mad_thresh=float(self.param_boxes[0].text()),
                min_area_pix=int(self.param_boxes[1].text()),
                max_area_pix=int(self.param_boxes[2].text()),
                min_duration_secs=float(self.param_boxes[3].text()),
                max_duration_secs=float(self.param_boxes[4].text())
            )

            flashes_in_block = detector.detect_in_block(frame_idx)

            flash_list.append(flashes_in_block)

        # Cleanup detector resources
        detector.cleanup()
        
        # Restore the detection button
        self.detect_button.setEnabled(True)
        self.detect_button.setText("Detect Flashes")

        QMessageBox.information(
            self, "Detection Complete",
            f"Detected {len(flash_list)} flashes"
        )

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
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.frame_slider.setEnabled(True)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(max(0, self.total_frames - 1))
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

        # Scale image to fit label while maintaining aspect ratio
        pixmap_scaled = pixmap.scaled(
            self.image_label.size(),
            QtCoreQt.AspectRatioMode.KeepAspectRatio,
            QtCoreQt.TransformationMode.SmoothTransformation
        )

        # Show the image in the label
        self.image_label.setPixmap(pixmap_scaled)

        # Update slider and frame number box
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_idx)
        self.frame_slider.blockSignals(False)
        self.frame_number_box.setText(f"Frame: {frame_idx}")

    def slider_frame_changed(self, value):
        self.show_frame(value)
