from turtle import pd
import cupy as cp
import cv2
import time

from build.lib.lifdetector import detection

# Detect if CUDA is available for CuPy
cuda_available = cp.cuda.is_available()

if cuda_available:

    print("CUDA is available")
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

                # Convert to grayscale if color (BGR)
                if len(frame_raw.shape) == 3 and frame_raw.shape[2] == 3:
                    frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)
                else:
                    frame = frame_raw

                # Append the frame (numpy array) to the list
                frames_list.append(frame)

            else:
                print(f"Short frame block - end of video reached at frame {f_start + frame_idx}")
                break
        
            # Convert frame list to 3D frame block (time, y, x)
            self.frame_block = npx.array(frames_list, dtype=npx.float32)
            self.n_rows, self.n_cols = self.frame_block.shape[1], self.frame_block.shape[2]
            
            # Calculate temporal median and MAD of noisy signal at each pixel
            s_tmed = npx.median(self.frame_block, axis=0)

            # t0 = time.perf_counter()
            s_tmad = npx.median(npx.abs(self.frame_block - s_tmed), axis=0)

            # Create zero MAD mask
            zero_mad_mask = s_tmad == 0

            # Dilate zero MAD mask using a local maximum filter over 5x5 neighborhood
            footprint = npx.ones((5, 5), dtype=npx.uint8)
            mad_mask = spndx.maximum_filter(zero_mad_mask.astype(npx.uint8), footprint=footprint) > 0

            # Create a 2D spatial threshold map from the temporal median and MAD
            thresh_map = s_tmed + self.mad_thresh * s_tmad

            # Smooth threshold map while ignoring MAD-masked pixels
            thresh_map = self.masked_gaussian_filter(thresh_map, mad_mask, sigma=5.0)

            # Reapply MAD mask to threshold map
            thresh_map = npx.where(mad_mask, npx.nan, thresh_map)

            # Identify suprathreshold pixels in the 3D frame block
            thresh_3d = npx.expand_dims(thresh_map, axis=0)
            suprathreshold = self.frame_block >= thresh_3d

            # Iteratively adjust threshold multiplier upwards until < 100 voxel detections found
            max_detections = 100
            thresh_scale = 1.0
            total_detections = npx.sum(suprathreshold)

            while total_detections > max_detections:
                thresh_scale += 0.1
                suprathreshold = self.frame_block >= thresh_3d * thresh_scale
                total_detections = npx.sum(suprathreshold)

            print(f"Final threshold scale factor: {thresh_scale:.2f}")
            print(f"Total candidate detections: {total_detections}")

            # Find connected components in 3D detection block
            t0 = time.perf_counter()
            labels, num_features = spndx.label(suprathreshold, structure=self._strel6)
            print(f"Connected components labeling computed in {(time.perf_counter() - t0)*1e3:.2f} ms")
            print(f"Number of detected components: {num_features}")

            anomaly_list = []

            for l in range(1, num_features + 1):

                # Find bounding box of component l
                component_indices = npx.where(labels == l)
                bb_t_min, bb_t_max = int(npx.min(component_indices[0])), int(npx.max(component_indices[0])+1)
                bb_y_min, bb_y_max = int(npx.min(component_indices[1])), int(npx.max(component_indices[1])+1)
                bb_x_min, bb_x_max = int(npx.min(component_indices[2])), int(npx.max(component_indices[2])+1)

                # Center of bounding box
                com_y, com_x = (bb_y_min + bb_y_max) / 2, (bb_x_min + bb_x_max) / 2

                # Calculate spatial area and temporal duration of component l
                area_pix = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
                duration_secs = (bb_t_max - bb_t_min) / self.fps

                candidate = (
                    area_pix >= self.min_area_pix and
                    area_pix <= self.max_area_pix and
                    duration_secs >= self.min_duration_secs and
                    duration_secs <= self.max_duration_secs
                )
                
                if candidate:

                    # Extract anomaly bounding box image data
                    anomaly_image = self.frame_block[
                        bb_t_min:bb_t_max,
                        bb_y_min:bb_y_max,
                        bb_x_min:bb_x_max
                    ]

                    this_anomaly = {
                        'label': l,
                        't_min': bb_t_min,
                        't_max': bb_t_max,
                        'y_min': bb_y_min,
                        'y_max': bb_y_max,
                        'x_min': bb_x_min,
                        'x_max': bb_x_max,
                        'com_x': com_x,
                        'com_y': com_y,
                        'area_pix': area_pix,
                        'duration_secs': duration_secs,
                        'anomaly_image': anomaly_image
                    }

                    anomaly_list.append(this_anomaly)

            self.detected_anomalies = anomaly_list

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