import numpy as np
import cv2
import os
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
        frame[y, x] = (255, 255, 255)  # White dots for all events

    return frame

def apply_dbscan(events, eps=5.0, min_samples=10):
    """
    Apply cuML DBSCAN clustering on raw event data.
    """
    if len(events) == 0:
        return np.array([]), np.array([])

    # Convert event data into a cuDF DataFrame and ensure float32 format
    data = cudf.DataFrame({
        'x': np.array([x for x, y, p, t in events], dtype=np.float32),
        'y': np.array([y for x, y, p, t in events], dtype=np.float32)
    })

    # Apply GPU-accelerated DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

    # Convert results to NumPy arrays for further processing
    labels = clustering.labels_.to_numpy()
    return labels, data.to_pandas().values  # Convert to NumPy format

def draw_clusters(frame, events, labels):
    """
    Draw DBSCAN clusters directly on the event frame.
    """
    unique_labels = set(labels)
    colors = {label: np.random.randint(0, 255, 3) for label in unique_labels if label != -1}

    for (x, y), label in zip(events, labels):
        if label == -1:
            color = (0, 0, 255)  # Red for noise points
        else:
            color = tuple(map(int, colors[label]))  # Random color for clusters
        
        cv2.circle(frame, (int(x), int(y)), 2, color, -1)

    return frame

def process_event_stream(input_path, output_dir):
    """
    Process event stream, apply GPU-accelerated DBSCAN clustering, and save frames.
    """
    mv_iterator = EventsIterator(input_path=input_path)
    height, width = mv_iterator.get_size()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, evs in enumerate(mv_iterator):
        if len(evs) == 0:
            continue

        # Step 1: Convert event data into an image
        event_frame = events_to_frame(evs, width, height)

        # Step 2: Apply cuML DBSCAN clustering
        labels, clustered_events = apply_dbscan(evs)

        # Step 3: Overlay clusters onto the event frame
        event_frame = draw_clusters(event_frame, clustered_events, labels)

        # Step 4: Save the frame as an image
        output_filename = os.path.join(output_dir, f"frame_{i:06d}.png")
        cv2.imwrite(output_filename, event_frame)

        print(f"Saved: {output_filename}")

if __name__ == "__main__":
    input_path = r"adap_0_bottle_big.raw"  # Update with your RAW file
    output_dir = "dbscan_frames"  # Change to a suitable path inside the container
    process_event_stream(input_path, output_dir)
