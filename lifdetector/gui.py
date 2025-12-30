import cv2
import numpy as np
import threading
import time

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
    QSizePolicy,
    QStyle
)
from PyQt6.QtGui import (
    QPixmap,
    QImage,
    QAction,
    QFont
)

from .detection import AnomalyDetector
from .review import ReviewCanvas


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
        self.displayed_frame = 0

        # Initialize the anomaly list
        self.anomaly_list = []
        self.n_anomalies = 0
        self.anomaly_idx = 0

    def setup_detection_ui(self):
        """
        Set up the UI components for the main Detection tab:
        1. AVI file selection
        2. Video display
        3. Frame slider and controls
        4. Detection button
        5. Quit button
        """

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

        # -------------------------
        # Playback control buttons
        # -------------------------

        button_size = 40
        button_font = QFont()
        button_font.setPointSize(16)

        # Play/Pause button
        self.playpause_button = QToolButton()
        playpause_style = QApplication.style()
        self.play_icon = playpause_style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.pause_icon = playpause_style.standardIcon(QStyle.StandardPixmap.SP_MediaPause)
        self.playpause_button.setIcon(self.play_icon)
        self.playpause_button.setIconSize(self.playpause_button.sizeHint())
        self.playpause_button.setFixedSize(button_size, button_size)
        self.playpause_button.clicked.connect(self.playpause_video)

        # Step +1 button
        self.plus1_button = QToolButton()
        step_plus1_style = QApplication.style()
        step_plus1_icon = step_plus1_style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward)
        self.plus1_button.setIcon(step_plus1_icon)
        self.plus1_button.setIconSize(self.plus1_button.sizeHint())
        self.plus1_button.setFixedSize(button_size, button_size)
        self.plus1_button.clicked.connect(self.step_forward)

        # Step -1 button
        self.minus1_button = QToolButton()
        step_minus1_style = QApplication.style()
        step_minus1_icon = step_minus1_style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward)
        self.minus1_button.setIcon(step_minus1_icon)
        self.minus1_button.setIconSize(self.minus1_button.sizeHint())
        self.minus1_button.setFixedSize(button_size, button_size)
        self.minus1_button.clicked.connect(self.step_backward)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.frame_number_box)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.minus1_button)
        controls_layout.addWidget(self.playpause_button)
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

        # Navigation controls for anomalies
        nav_layout = QHBoxLayout()
        self.prev_anomaly_button = QToolButton()
        self.prev_anomaly_button.setText('Previous')
        self.prev_anomaly_button.clicked.connect(self.show_prev_anomaly)
        self.next_anomaly_button = QToolButton()
        self.next_anomaly_button.setText('Next')
        self.next_anomaly_button.clicked.connect(self.show_next_anomaly)
        nav_layout.addWidget(self.prev_anomaly_button)
        nav_layout.addWidget(self.next_anomaly_button)
        review_layout.addLayout(nav_layout)

        self.review_canvas = ReviewCanvas(parent=self.review_tab)
        review_layout.addWidget(self.review_canvas, stretch=1)

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

        self.current_anomaly_index = 0
        self.update_review_buttons()

    def update_review_buttons(self):
        n = len(self.anomaly_list) if hasattr(self, 'anomaly_list') else 0
        self.prev_anomaly_button.setEnabled(n > 1 and self.current_anomaly_index > 0)
        self.next_anomaly_button.setEnabled(n > 1 and self.current_anomaly_index < n - 1)

    def show_prev_anomaly(self):
        if hasattr(self, 'anomaly_list') and self.current_anomaly_index > 0:
            self.current_anomaly_index -= 1
            self.display_current_anomaly()
            self.update_review_buttons()

    def show_next_anomaly(self):
        if hasattr(self, 'anomaly_list') and self.current_anomaly_index < len(self.anomaly_list) - 1:
            self.current_anomaly_index += 1
            self.display_current_anomaly()
            self.update_review_buttons()

    def display_current_anomaly(self):
        if hasattr(self, 'anomaly_list') and self.anomaly_list:
            anomaly = self.anomaly_list[self.current_anomaly_index]
            self.review_canvas.plot_anomaly(anomaly, cuda_available=False)

    def setup_configuration_ui(self):

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
    
    def playpause_video(self):
        """
        Toggle play/pause state of video playback
        """
        if not self.cap:
            return

        # Toggle play/pause state
        self._playing = not getattr(self, '_playing', False)

        if self._playing:
            self.playpause_button.setIcon(self.pause_icon)
        else:
            self.playpause_button.setIcon(self.play_icon)

        def run():
            while self._playing and self.cap:
                pos = self.displayed_frame + 1
                if pos >= self.total_frames:
                    self._playing = False
                    self.playpause_button.setIcon(self.play_icon)
                    break
                self.show_frame(pos)
                if not self._playing:
                    break
                if self.displayed_frame >= self.total_frames - 1:
                    break
                time.sleep(1.0 / max(1, self.cap.get(cv2.CAP_PROP_FPS)))
        
        threading.Thread(target=run, daemon=True).start()

    def step_forward(self):
        if not self.cap:
            return
        pos = min(self.displayed_frame + 1, self.total_frames - 1)
        self.show_frame(pos)

    def step_backward(self):
        if not self.cap:
            return
        pos = max(0, self.displayed_frame - 1)
        self.show_frame(pos)

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

        # Init grand list of anomaly dictionaries
        grand_anomaly_list = []

        # Hardwire block size to 32 frames for now
        block_size = 32

        # Create the detector with parameters from the UI
        detector = AnomalyDetector(
            video_path=self.video_path,
            mad_thresh=float(self.param_boxes[0].text()),
            min_area_pix=int(self.param_boxes[1].text()),
            max_area_pix=int(self.param_boxes[2].text()),
            min_duration_secs=float(self.param_boxes[3].text()),
            max_duration_secs=float(self.param_boxes[4].text())
        )

        # Loop over blocks of frames
        for frame_idx in range(0, self.total_frames, block_size):

            print(f"\nProcessing block starting at frame {frame_idx}...")

            self.show_frame(frame_idx)
            self.frame_number_box.setText(f"Frame: {frame_idx}")
            QApplication.processEvents()

            # Run detection on the current block starting at frame_idx
            # Detected candidate flash anomalies are added to detector.anomaly_list
            detector.detect_in_block(start_frame=frame_idx, block_size=block_size)

            # Update running anomaly count in the detection count box
            current_count = len(detector.anomaly_list)
            self.detection_count_box.setText(f"Detections: {current_count}")
            QApplication.processEvents()

        # Reset frame counter to 0
        self.show_frame(0)

        QMessageBox.information(
            self, "Detection Complete",
            f"Detected {len(detector.anomaly_list)} flashes"
        )

        # Pass detected anomalies to main window and review canvas
        self.anomaly_list = detector.anomaly_list

        # Set the current anomaly index to 0 and display
        self.current_anomaly_index = 0
        self.display_current_anomaly()
        self.update_review_buttons()

        # Cleanup detector resources
        detector.cleanup()

        # Restore the detection button
        self.detect_button.setEnabled(True)
        self.detect_button.setText("Detect Flashes")

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
        
        self.displayed_frame = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.displayed_frame)
        
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
        self.frame_slider.setValue(self.displayed_frame)
        self.frame_slider.blockSignals(False)
        self.frame_number_box.setText(f"Frame: {self.displayed_frame}")

    def slider_frame_changed(self, value):
        self.show_frame(value)


