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

        # Create horizontal montage of XY slices through anomaly ROI
        montage_frames = []
        for frame_idx in range(0, roi_signal_3d.shape[0]):
            slice_xy = roi_signal_3d[frame_idx, :, :]
            montage_frames.append(slice_xy)

        # Extract flash MIP profile
        flash_profile = npx.max(npx.max(roi_signal_3d, axis=2), axis=1)

        # Prepare data for plotting
        plt_t = roi_t_vec_s
        plt_flash_profile = flash_profile
        plt_montage = npx.array(montage_frames)

        # Plot anomaly results in space and time
        # Plot individual frames in as separate axes in row one
        # Plot anomaly temporal profile in row two, spanning all columns

        # Clear previous figure
        self.figure.clf()

        # First row: Montage of anomaly frames
        ax1 = self.figure.add_subplot(211)

        # Create list of subfigures for each row
        gridspec = ax1.get_subplotspec().get_gridspec()
        subfigs = [self.figure.add_subfigure(gs) for gs in gridspec]

        # Add subplots to first row
        subfigs[0].suptitle(f'Flash Onset at Frame {f_anomaly_start} (X: {com_x:0.1f}, Y: {com_y:0.1f})')

        # Create n_roi_frame subplots in top row subfigure
        n_montage_frames = len(montage_frames)

        # Get grand scaling max for montage frames
        grand_max = npx.max(plt_montage)

        axs_toprow = subfigs[0].subplots(nrows=1, ncols=n_montage_frames)
        for col, ax in enumerate(axs_toprow):
            ax.imshow(plt_montage[col], cmap='magma', vmax=grand_max)
            ax.axis('off')
            # Set title to time in seconds
            ax.set_title(f'{plt_t[col]:.2f} s')

        ax_bottomrow = subfigs[1].subplots(nrows=1, ncols=1)
        ax_bottomrow.plot(plt_t, plt_flash_profile)

        # Add transparent green box to indicate detected flash duration
        # ax_bottomrow.axvspan(float(bb_t_min)/fps, float(bb_t_max)/fps, color='green', alpha=0.2)

        # Add box markers for frames shown in montage
        for frame_idx in range(0, roi_signal_3d.shape[0]):
            ax_bottomrow.axvline(plt_t[frame_idx], color='red', linestyle='--', alpha=0.5)

        ax_bottomrow.set_xlabel('Time (s)')
        ax_bottomrow.set_ylabel('Temporal MIP')
        ax_bottomrow.set_title(f'Temporal MIP for Anomaly {label} (Area: {area_px} px, Duration: {duration_s:.2f} s)')

        self.draw()



        