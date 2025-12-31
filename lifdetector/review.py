import matplotlib.pyplot as plt
import numpy as npx

# Matplotlib imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


class ReviewCanvas(FigureCanvas):
    """
    Derived Matplotlib canvas for displaying review plots in the Review tab
    """

    def __init__(self, parent=None):

        self.fig_dpi = 100

        # Get size of parent widget for figure sizing
        if parent is not None:
            self.fig_width = parent.width() / self.fig_dpi  # Convert to inches assuming 100 dpi
            self.fig_height = parent.height() / self.fig_dpi  # Convert to inches assuming 100 dpi
        else:
            self.fig_width = 8  # Default width in inches
            self.fig_height = 6  # Default height in inches

        fig = Figure(figsize=(self.fig_width, self.fig_height), dpi=self.fig_dpi)

        super().__init__(fig)
        self.setParent(parent)

        # Initialize canvas with blank plot
        self.plot_blank()

    def plot_blank(self):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "No anomalies detected", ha='center', va='center', fontsize=16)
        ax.axis('off')
        self.draw()

    def plot_anomaly(self, anomaly, cuda_available=False):
        """
        Plot a subplot figure with the following and display in the MatplotlibCanvas in the review tab
        1) Extracted timestamp from anomaly flash frame
        2) Top row: Horizontal montage of XY slices through flash region center for two frames before and after flash onset
        3) Bottom row: Temporal MIP profile of flash region with detected flash duration highlighted
        
        :param anomaly: Anomaly dictionary
        :param cuda_available: Flag indicating if CUDA is available
        """

        # Extract relevant data from anomaly
        label = anomaly['label']
        f_anomaly_start = anomaly['f_anomaly_start']  # Anomaly start frame in original video
        roi_f_anomaly_start = anomaly['roi_f_anomaly_start']  # Anomaly start frame within ROI
        roi_t_vec_s = anomaly['roi_t_vec_s']  # Time vector for ROI in seconds relative to ROI start
        duration_s = anomaly['duration_s']  # Anomaly duration in seconds
        area_px = anomaly['area_px']  # Anomaly area in pixels
        com_x = anomaly['com_x']  # Anomaly centroid X in original frame coordinates
        com_y = anomaly['com_y']  # Anomaly centroid Y in original frame coordinates
        roi_signal_3d = anomaly['roi_signal_3d']  # Extracted ROI signal data S_roi(t, x, y)
        roi_suprathresh_3d = anomaly['roi_suprathresh_3d']  # Suprathreshold mask for ROI
        anomaly_tprofile = anomaly['anomaly_tprofile']  # Temporal profile of anomaly (signal MIP onto time axis)

        # Create horizontal montage of XY slices through anomaly ROI
        signal_montage = []
        supra_montage = []
        n_roi_frames = roi_signal_3d.shape[0]
        for frame_idx in range(0,n_roi_frames):
            signal_montage.append(roi_signal_3d[frame_idx, :, :])
            supra_montage.append(roi_suprathresh_3d[frame_idx, :, :])

        # Concatenate montage frames into single array for plotting
        plt_signal_montage = npx.concatenate(signal_montage, axis=1)
        plt_supra_montage = npx.concatenate(supra_montage, axis=1)

        # Plot anomaly results in space and time
        # Plot individual frames in as separate axes in row one
        # Plot anomaly temporal profile in row two, spanning all columns

        # Clear previous figure
        self.figure.clf()

        # First row: Montage of anomaly signal frames
        ax1 = self.figure.add_subplot(311)
        ax1.imshow(plt_signal_montage, cmap='magma')
        ax1.set_title(f'Anomaly {label} Signal (Frame start: {f_anomaly_start} px, Duration: {duration_s:.2f} s)')
        ax1.axis('off')

        # Second row: Montage of anomaly suprathreshold mask frames
        ax2 = self.figure.add_subplot(312)
        ax2.imshow(plt_supra_montage, cmap='gray')
        ax2.set_title(f'Anomaly {label} Suprathreshold Mask Montage (Area: {area_px} px, Duration: {duration_s:.2f} s)')
        ax2.axis('off')

        # Third row: Temporal MIP profile of anomaly
        ax3 = self.figure.add_subplot(313)
        ax3.plot(roi_t_vec_s, anomaly_tprofile, color='blue')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Temporal MIP')
        ax3.set_title(f'Temporal MIP for Anomaly {label} (Area: {area_px} px, Duration: {duration_s:.2f} s)')

        # Add transparent green box to indicate detected flash duration
        # ax_bottomrow.axvspan(float(bb_t_min)/fps, float(bb_t_max)/fps, color='green', alpha=0.2)

        # Add box markers for frames shown in montage
        for frame_idx in range(0, n_roi_frames):
            ax3.axvline(roi_t_vec_s[frame_idx], color='red', linestyle='--', alpha=0.5)

        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Temporal MIP')
        ax3.set_title(f'Temporal MIP for Anomaly {label} (Area: {area_px} px, Duration: {duration_s:.2f} s)')

        self.draw()



        