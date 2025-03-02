import os
import shutil

# Set the base directory (change this if your folders are not in the current directory)
base_dir = os.getcwd()

# Define destination directories
dest_base = os.path.join(base_dir, "segmentation_new")
dest_images = os.path.join(dest_base, "images")
dest_masks = os.path.join(dest_base, "masks")

# Create destination directories if they don't exist
for directory in [dest_base, dest_images, dest_masks]:
    os.makedirs(directory, exist_ok=True)

# Loop through the folders in the base directory
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        # Process event_frames_* folders for images
        if folder.startswith("event_frames_"):
            # Remove the prefix "event_frames_"
            folder_suffix = folder[len("event_frames_"):]
            for file in os.listdir(folder_path):
                if file.endswith(".png"):
                    src = os.path.join(folder_path, file)
                    # Rename by prefixing with the folder name (without the prefix) to avoid clashes
                    new_filename = f"{folder_suffix}_{file}"
                    dst = os.path.join(dest_images, new_filename)
                    print(f"Moving {src} to {dst}")
                    shutil.move(src, dst)
        # Process dbscan_frames_* folders for masks
        elif folder.startswith("dbscan_frames_"):
            # Remove the prefix "dbscan_frames_"
            folder_suffix = folder[len("dbscan_frames_"):]
            for file in os.listdir(folder_path):
                if file.endswith(".png"):
                    src = os.path.join(folder_path, file)
                    new_filename = f"{folder_suffix}_{file}"
                    dst = os.path.join(dest_masks, new_filename)
                    print(f"Moving {src} to {dst}")
                    shutil.move(src, dst)

print("Files moved successfully!")
