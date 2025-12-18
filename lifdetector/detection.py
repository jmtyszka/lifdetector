
import numpy as np
import cv2

def detect_flashes(
        video_path:str,
        start_frame:int=0,
        frame_window:int=8,
        z_thresh:float=3.0,
        min_area:int=5,
    ) ->list[int]:
    """
    Detect short, small, bright flashes in a video.
    Returns a list of frame indices where flashes are detected.
    Args:
        video_path: Path to the AVI file.
        start_frame: Frame index to start processing from.
        frame_window: Number of frames to consider for temporal isolation.
        z_thresh: Z-score threshold for flash detection.
        min_area: Minimum area (in pixels) for a flash.
    """

    cap = cv2.VideoCapture(video_path)
    
    # Check if the video source was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    # Load frame_window frames to process from start_frame to start_frame + frame_window
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    f_start = max(0, start_frame)
    f_end = min(total_frames, f_start + frame_window)
    n_frames = f_end - f_start

    # Fast forward to f_start
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    # 1. Initialize a list to hold frames
    frames_list = []

    # 2. Read frames in a loop
    for frame_idx in range(n_frames):

        # ret: boolean, True if frame was read successfully
        # frame: the image frame as a NumPy array
        ret, frame = cap.read()

        if ret:
            # Append the frame (numpy array) to the list
            frames_list.append(frame)
        else:
            print(f"Warning: Could not read frame {frame_idx + 1}. Breaking loop.")
            break

    # 3. Release the capture object
    cap.release()

    n_frames_loaded = len(frames_list)
    print(f"Loaded {n_frames_loaded} frames from {video_path} (from frame {f_start} to {f_start + n_frames_loaded - 1})")

    # 4. Convert the list of frames into a single NumPy array
    if frames_list:

        # Shape will be (num_frames, height, width, channels)
        frames_array = np.array(frames_list)

    # Detected flash list
    flashes = 5  # Placeholder for detected flash frame indices

    #     # Convert frame to grayscale
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     # Threshold for bright spots
    #     _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    #     # Find contours of bright spots
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     flash_found = False
    #     for cnt in contours:
    #         area = cv2.contourArea(cnt)
    #         if area >= min_area:
    #             flash_found = True
    #             break
    #     # Check for temporal isolation (flash should not persist in previous frames)
    #     if flash_found:
    #         is_isolated = True
    #         for prev in prev_frames:
    #             _, prev_mask = cv2.threshold(prev, threshold, 255, cv2.THRESH_BINARY)
    #             if np.any(prev_mask):
    #                 is_isolated = False
    #                 break
    #         if is_isolated:
    #             flashes.append(frame_idx)
    #     prev_frames.append(gray)
    #     if len(prev_frames) > window:
    #         prev_frames.pop(0)
    #     frame_idx += 1
    #     ret, frame = cap.read()

    # cap.release()
    
    return flashes
