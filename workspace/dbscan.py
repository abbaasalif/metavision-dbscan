import numpy as np
import cv2
import os
import random
import cudf
from cuml.cluster import DBSCAN
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_core.event_io import EventsIterator

def events_to_frame(events, width, height):
    """
    Convert event data into an image frame for visualization.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for x, y, p, t in events:
        frame[y, x] = (255, 255, 255)  # White dot for each event
    return frame

def apply_dbscan(events, eps=5.0, min_samples=30):
    """
    Apply GPU-accelerated DBSCAN clustering on raw event data.
    """
    if len(events) == 0:
        return np.array([]), np.array([])

    # Convert event data into a cuDF DataFrame with x and y coordinates
    data = cudf.DataFrame({
        'x': np.array([x for x, y, p, t in events], dtype=np.float32),
        'y': np.array([y for x, y, p, t in events], dtype=np.float32)
    })

    # Run DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

    # Convert labels and event coordinates to NumPy arrays for later use
    labels = clustering.labels_.to_numpy()
    return labels, data.to_pandas().values  # Returns a NumPy array of [x, y]

def draw_clusters(frame, events, labels):
    """
    Overlay clusters on the frame.
    Noise points are drawn in red and all other cluster points in white.
    """
    for (x, y), label in zip(events, labels):
        if label == -1:
            color = (0, 0, 255)  # Red for noise points
        else:
            color = (255, 255, 255)  # White for cluster points
        cv2.circle(frame, (int(x), int(y)), 2, color, -1)
    return frame

def process_event_stream(input_path, output_dir, num_frames=25):
    """
    Process event stream from the RAW file by first collecting all event frames,
    then randomly selecting num_frames to apply DBSCAN clustering and saving them.
    Also saves the original event frame (without clustering overlay) in a separate folder.
    """
    mv_iterator = EventsIterator(input_path=input_path)
    height, width = mv_iterator.get_size()

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    original_frames_dir = "event_frames"
    os.makedirs(original_frames_dir, exist_ok=True)

    # Collect all event frames (each frame is a list/array of events)
    event_frames = []
    for evs in mv_iterator:
        if len(evs) == 0:
            continue
        event_frames.append(evs)

    # Randomly select up to num_frames from the collected frames
    num_to_select = min(num_frames, len(event_frames))
    selected_frames = random.sample(event_frames, num_to_select)

    # Process each selected frame
    for i, evs in enumerate(selected_frames):
        # Create the original event frame (without cluster overlay)
        original_frame = events_to_frame(evs, width, height)
        # Save the original event frame in the "event_frames" folder
        original_filename = os.path.join(original_frames_dir, f"frame_{i:06d}.png")
        cv2.imwrite(original_filename, original_frame)
        print(f"Saved original event frame: {original_filename}")

        # Apply DBSCAN clustering on the event data
        labels, clustered_events = apply_dbscan(evs)
        # Overlay cluster information onto a copy of the original frame
        cluster_frame = draw_clusters(original_frame.copy(), clustered_events, labels)
        # Save the clustered frame in the designated output directory
        output_filename = os.path.join(output_dir, f"frame_{i:06d}.png")
        cv2.imwrite(output_filename, cluster_frame)
        print(f"Saved DBSCAN frame: {output_filename}")

if __name__ == "__main__":
    input_path = r"mirror_amb_static.raw"  # Update with your RAW file path
    output_dir = "dbscan_frames"           # Update to your desired output directory for clustered frames
    process_event_stream(input_path, output_dir, num_frames=25)
