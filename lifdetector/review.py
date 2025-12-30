import matplotlib.pyplot as plt
import numpy as npx

# Matplotlib imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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

        # Clear previous figure
        self.figure.clf()

        # Extract relevant data from anomaly
        l = anomaly['label']
        bb_t_min = anomaly['t_min']
        bb_t_max = anomaly['t_max']
        bb_y_min = anomaly['y_min']
        bb_y_max = anomaly['y_max']
        bb_x_min = anomaly['x_min']
        bb_x_max = anomaly['x_max']
        com_x = anomaly['com_x']
        com_y = anomaly['com_y']

        # Time vector in seconds for bounding box
        bb_t_secs = npx.arange(bb_t_min, bb_t_max) / fps

        # Extract small region around flash CoM
        hw = 10  # Half width of region
        roi_x_min = int(max(com_x - hw, 0))
        roi_x_max = int(min(com_x + hw, signal_3d.shape[2]))
        roi_y_min = int(max(com_y - hw, 0))
        roi_y_max = int(min(com_y + hw, signal_3d.shape[1]))
        flash_region = signal_3d[:, roi_y_min:roi_y_max, roi_x_min:roi_x_max]

        # Create horizontal montage of XY slices through flash region center for two frames before and after flash onset
        montage_t_start = max(bb_t_min - 2, 0)
        montage_t_end = min(bb_t_max + 2, signal_3d.shape[0])
        montage_frames = []
        for frame_idx in range(montage_t_start, montage_t_end):
            slice_xy = flash_region[frame_idx, :, :]
            montage_frames.append(slice_xy)

        # Extract flash MIP profile
        flash_profile = npx.max(npx.max(flash_region, axis=2), axis=1)

        if cuda_available:
            plt_t = t.get()
            plt_flash_profile = flash_profile.get()
            plt_montage = npx.array(montage_frames).get()
        else:
            plt_t = t
            plt_flash_profile = flash_profile
            plt_montage = npx.array(montage_frames)

        # Plot anomaly results in space and time
        # Plot individual frames in as separate axes in row one
        # Plot anomaly temporal profile in row two, spanning all columns

        # Create 2x1 subplots
        fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize=(8, 6))

        # Clear subplots
        for ax in axs:
            ax.remove()

        # Create list of subfigures for each row
        gridspec = axs[0].get_subplotspec().get_gridspec()
        subfigs = [fig.add_subfigure(gs) for gs in gridspec]

        # Add subplots to first row
        subfigs[0].suptitle(f'Anomaly {l} Frames')

        # Create n_roi_frame subplots in top row subfigure
        n_montage_frames = len(montage_frames)

        # Get grand scaling max for montage frames
        grand_max = npx.max(plt_montage)

        axs_toprow = subfigs[0].subplots(nrows=1, ncols=n_montage_frames)
        for col, ax in enumerate(axs_toprow):
            ax.imshow(plt_montage[col], cmap='magma', vmax=grand_max)
            ax.axis('off')
            # Set title to time in seconds
            ax.set_title(f'{plt_t[montage_t_start + col]:.2f} s')

        ax_bottomrow = subfigs[1].subplots(nrows=1, ncols=1)
        ax_bottomrow.plot(plt_t, plt_flash_profile)

        # Add transparent green box to indicate detected flash duration
        ax_bottomrow.axvspan(float(bb_t_min)/fps, float(bb_t_max)/fps, color='green', alpha=0.2)

        # Add box markers for frames shown in montage
        for frame_idx in range(montage_t_start, montage_t_end):
            ax_bottomrow.axvline(plt_t[frame_idx], color='red', linestyle='--', alpha=0.5)

        ax_bottomrow.set_xlabel('Time (s)')
        ax_bottomrow.set_ylabel('Temporal MIP')
        ax_bottomrow.set_title(f'Temporal MIP for Anomaly {l}')

        plt.show()



        