import cupy as cp
import cv2
import time
import pandas as pd

from sklearn.mixture import GaussianMixture
from skimage.morphology import disk
from skimage.measure import regionprops

from build.lib.lifdetector import detection

# Detect if CUDA is available for CuPy
try:
    cuda_available = cp.cuda.is_available()
except Exception:
    cuda_available = False

if cuda_available:

    print("CUDA is available. Using GPU compute")
    import cupy as npx
    import cupyx.scipy.ndimage as spndx

    print(npx.show_config())

else:
    
    print("CUDA is not available. Using CPU compute")
    import numpy as npx
    import scipy.ndimage as spndx


class AnomalyDetector:
    """
    Class for performing anomaly detection on video frames block
    """

    def __init__(
            self,
            video_path:str,
            mad_thresh:float=5.0,
            min_area_pix:int=5,
            max_area_pix:int=100,
            min_duration_secs:float=0.05,
            max_duration_secs:float=0.25
        ):
        """
        Initialize the AnomalyDetector with a block of frames.
        Args:
            video_path: Path to the AVI file.
            mad_thresh: MAD threshold for flash detection.
        """

        self.verbose = False

        self.video_path = video_path
        self.n_rows = None
        self.n_cols = None
        self.block_size = 32  # Number of frames per block
        self.frame_block = None  # Placeholder for frame block data

        # 3D structuring element with 6-connectivity
        self._strel6 = spndx.generate_binary_structure(3, 1)
        
        # Detection limits
        self.mad_thresh = mad_thresh
        self.min_area_pix = min_area_pix
        self.max_area_pix = max_area_pix
        self.min_duration_secs = min_duration_secs
        self.max_duration_secs = max_duration_secs

        # Initialize list to hold detected anomalies
        self.detected_anomalies = []

        # Open video capture
        self.cap = cv2.VideoCapture(video_path)
    
        # Check if the video source was opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open video source.")
            exit()

        # Video properties
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def detect_in_block(self, start_frame:int=0):
        """
        Detect anomalies (flashes) in the frame block.
        Record anomalies as Anomaly objects containing timing and bounding box metadata
        and extracted image data surrounding the anomaly.

        Args:
            frame_start: Starting frame index of block in the original video.
        """

        # Adjust block length if it exceeds total frames
        f_start = max(0, start_frame)
        f_end = min(self.total_frames, f_start + self.block_size)
        n_frames = f_end - f_start

        # Fast forward to f_start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

        # Initialize frame block
        frames_list = []

        # Load frame block
        for frame_idx in range(n_frames):

            # ret: boolean, True if frame was read successfully
            # frame: the image frame as a NumPy array
            ret, frame_raw = self.cap.read()
            if ret:
                # Append the frame (numpy array) to the list
                frames_list.append(self.safe_grayscale(frame_raw))
            else:
                print(f"Short frame block - end of video reached at frame {f_start + frame_idx}")
                break
        
            # Convert frame list to 3D signal block (time, y, x)
            self.signal_3d = npx.array(frames_list, dtype=npx.float32)
            self.n_rows, self.n_cols = self.signal_3d.shape[1], self.signal_3d.shape[2]

            # Phase 1: Statistical anomaly detection
            # - Calculate temporal mean and SD of noisy signal at each pixel
            # - Create a pixel exclusion mask based on tSD
            # - Create a smoothed mask smoothed detection threshold map
            # - Identify suprathreshold pixels in 3D frame block using threshold map
            
            # Calculate temporal mean and SD of noisy signal at each pixel
            self.s_tmean = npx.mean(self.signal_3d, axis=0)
            self.s_tsd = npx.std(self.signal_3d, axis=0)

            # Create pixel exclusion mask where tSD close to zero or very high
            # Corresponding to clipped pixels or motion artifacts close to clipped regions
            self.create_exclusion_mask()

            # Calculate mask-smoothed detection threshold map
            self.create_threshold_map()

            # Identify suprathreshold voxels in 3D frame block
            self.identify_suprathreshold_voxels()

            # Phase 2: Shortlist anomalies using connected components analysis
            # - Apply user-defined criteria to size and duration of candidate regions
            self.shortlist_anomalies()

    def create_exclusion_mask(self):
        """
        Create pixel exclusion mask based on temporal SD.
        Exclude pixels with very low or very high temporal SD.
        """

        # Gaussian mixture model to SD histogram for pixel exclusion thresholding

        # Subsample full SD distribution for GMM fitting if too large
        max_pixels = 5000
        if self.s_tsd.size > max_pixels:
            step = int((self.s_tsd.size / max_pixels) ** 0.5)
            s_tsd_sub = self.s_tsd[::step, ::step]
            if self.verbose:
                print(f"Subsampling tSD for GMM fitting: step={step}, subsampled pixels={s_tsd_sub.size}")
            sd_values = s_tsd_sub.flatten().reshape(-1, 1)
        else:
            sd_values = self.s_tsd.flatten().reshape(-1, 1)

        if self.verbose:
            print("Number of pixels:", sd_values.shape[0])

        model = GaussianMixture(n_components=2, covariance_type='full', random_state=0)

        model.fit(sd_values.get() if cuda_available else sd_values)
        weights = model.weights_.flatten()
        means = model.means_.flatten()
        sds = model.covariances_.flatten() ** 0.5

        # Select larger gaussian component as sensor noise
        noise_component = means.argmax()
        clipped_component = 1 - noise_component

        w_clipped = weights[clipped_component]
        m_clipped = means[clipped_component]
        sd_clipped = sds[clipped_component]

        w_noise = weights[noise_component]
        m_noise = means[noise_component]
        sd_noise = sds[noise_component]

        if self.verbose:
            print(f"GMM Clipped Component: Weight={w_clipped:.2f}, Mean={m_clipped:.2f}, SD={sd_clipped:.2f}")
            print(f"GMM Noise Component: Weight={w_noise:.2f}, Mean={m_noise:.2f}, SD={sd_noise:.2f}")

        # Use the naive Bayes discrimination boundary as the exclusion threshold where P(noise) = P(clipped)
        #
        # Reference:
        # Duda, R. O., Hart, P. E., & Stork, D. G. (2001). Pattern Classification (2nd Edition),
        # Section 2.5.2: "Normal Density: Discriminant Functions for the Normal Density". Wiley-Interscience.
        
        nbd = (
            ((m_clipped**2) - (m_noise**2)) + 
            2 * (sd_clipped**2 * npx.log(w_noise * sd_clipped) -
                sd_noise**2 * npx.log(w_clipped * sd_noise))
        ) / (2 * (m_clipped - m_noise))

        if self.verbose:
            print(f"\nNaive Bayes Discrimination Boundary (tSD): {nbd:.2f}")

        # Lower threshold avoids clipped voxels
        tsd_thresh_low = nbd

        # Upper threshold to exclude unusually high tSD pixels but cap at 255
        tsd_thresh_high = m_noise + 10 * sd_noise
        if tsd_thresh_high > 255:
            tsd_thresh_high = 255.0

        if self.verbose:
            print(f"\ntSD Exclusion thresholds: Low={tsd_thresh_low:.2f}, High={tsd_thresh_high:.2f}")

        # Exclude clipped pixels with tSD below threshold
        exclude_mask = (self.s_tsd < tsd_thresh_low) | (self.s_tsd > tsd_thresh_high)

        # Dilate mask using a k-pixel diameter circular structured element
        # Set k in UI config tab
        r_disk = 11
        footprint = disk(r_disk).astype(npx.uint8)
        if self.verbose:
            print(f'Structured element radius: {r_disk}')

        self.exclude_mask = spndx.maximum_filter(exclude_mask.astype(npx.uint8), footprint=footprint) > 0

    def create_threshold_map(self):
        """
        Create smoothed detection threshold map using masked Gaussian filtering.
        """

        # Create a 2D spatial threshold map from the temporal mean and sd
        alpha = 5.0  # Threshold scaling factor
        self.thresh_map = self.s_tmean + alpha * self.s_tsd

        # Add pixels with thrshold > 255 to exclusion mask
        self.exclude_mask = self.exclude_mask | (self.thresh_map > 255)

        # Apply masked Gaussian filtering to threshold map
        # Using sigma=10.0 as an example; set from UI config tab
        t0 = time.perf_counter()
        self.thresh_map = self.masked_gaussian_filter(self.thresh_map, self.exclude_mask, sigma=10.0)
        if self.verbose:
            print(f"Masked Gaussian filtering computed in {(time.perf_counter() - t0)*1e3:.2f} ms")

        # Reapply exclusion mask to threshold map
        self.thresh_map = npx.where(self.exclude_mask, npx.nan, self.thresh_map)

    def identify_suprathreshold_voxels(self):
        """
        Identify suprathreshold voxels in 3D frame block using threshold map.
        """
        # Threshold the signal to create candidate anomaly detections
        thresh_map_3d = npx.expand_dims(self.thresh_map, axis=0)
        self.suprathreshold_3d = self.signal_3d >= thresh_map_3d

        # Display temporal sum of suprathreshold voxels
        self.detection_tsum = npx.sum(self.suprathreshold_3d.astype(npx.uint8), axis=0)

    def shortlist_anomalies(self):
        """
        Shortlist anomalies using connected components analysis and user-defined criteria.
        """

        # Binarize temporal sum image for morphological analysis
        detection_bin = self.detection_tsum > 0

        labels, num_labels = spndx.label(detection_bin)
        if self.verbose:
            print(f"Number of initial candidate regions: {num_labels}")
        regions = regionprops(labels.get() if cuda_available else labels)

        anomaly_list = []

        for region in regions:

            # Get XY bounding box of region
            bb_xy = region.bbox

            # Extract bounding box subregion from suprathreshold 3D mask
            supra_bb_xy = self.suprathreshold_3d[:, bb_xy[0]:bb_xy[2], bb_xy[1]:bb_xy[3]]

            # Get temporal bounding box
            supra_indices = npx.where(supra_bb_xy)
            t_min, t_max = int(npx.min(supra_indices[0])), int(npx.max(supra_indices[0])+1)

            # Duration of temporal bounding box in seconds
            duration = (t_max - t_min) / self.fps

            # Apply phase 2 candidate criteria
            candidate = duration > 0.05 and duration < 0.20 and region.area >= 5 and region.area <= 20

            if candidate:
                anomaly_list.append({
                    'label': region.label,
                    'x_min': bb_xy[1],
                    'x_max': bb_xy[3],
                    'y_min': bb_xy[0],
                    'y_max': bb_xy[2],
                    't_min': t_min,
                    't_max': t_max,
                    'duration_s': duration,
                    'area_px': region.area,
                    'com_x': region.centroid[1],
                    'com_y': region.centroid[0],
                })

        # Convert to dataframe for easier viewing
        anomaly_df = pd.DataFrame(anomaly_list)

        if self.verbose:
            print(anomaly_df)

    @staticmethod
    def safe_grayscale(frame):
        """
        Safely convert frame to grayscale.
        """

        # Convert to grayscale if color (BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        return gray_frame

    def masked_gaussian_filter(self, img, mask, sigma):
                              
        """
        Apply Gaussian filter to image while ignoring masked values.
        Args:
            img: 2D array to be filtered.
            mask: 2D boolean array where True indicates masked (ignored) pixels.
            sigma: Standard deviation for Gaussian kernel.
        """
        image_filled = npx.where(mask, 0, img)
        filtered_image = spndx.gaussian_filter(image_filled, sigma=sigma)
        normalization = spndx.gaussian_filter((~mask).astype(npx.float32), sigma=sigma)
        normalization = npx.where(normalization == 0, npx.nan, normalization)
        return filtered_image / normalization
    
    def cleanup(self):
        """
        Release video capture resources.
        """
        self.cap.release()