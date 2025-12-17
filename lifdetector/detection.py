
import numpy as np
import cv2

def detect_flashes(video_path, threshold=220, min_area=5, window=1, progress_callback=None):
    """
    Detect short, small, bright flashes in a video.
    Returns a list of frame indices where flashes are detected.
    Args:
        video_path: Path to the AVI file.
        threshold: Brightness threshold for flash detection.
        min_area: Minimum area (in pixels) for a flash.
        window: Number of frames to consider for temporal isolation.
    """
    cap = cv2.VideoCapture(video_path)
    flashes = []
    prev_frames = []
    frame_idx = 0
    ret, frame = cap.read()
    while ret:
        if progress_callback is not None:
            progress_callback(frame_idx)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Threshold for bright spots
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        # Find contours of bright spots
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flash_found = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                flash_found = True
                break
        # Check for temporal isolation (flash should not persist in previous frames)
        if flash_found:
            is_isolated = True
            for prev in prev_frames:
                _, prev_mask = cv2.threshold(prev, threshold, 255, cv2.THRESH_BINARY)
                if np.any(prev_mask):
                    is_isolated = False
                    break
            if is_isolated:
                flashes.append(frame_idx)
        prev_frames.append(gray)
        if len(prev_frames) > window:
            prev_frames.pop(0)
        frame_idx += 1
        ret, frame = cap.read()
    cap.release()
    return flashes
