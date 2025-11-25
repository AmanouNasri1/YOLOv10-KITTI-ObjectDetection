import os
from glob import glob
from ultralytics import YOLO

def rename_files(folder_path, extensions):
    """Rename all files in folder_path with given extensions sequentially starting from 0."""
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(folder_path, f"*.{ext}")))
    
    files.sort()  # ensures consistent order
    for idx, file_path in enumerate(files):
        ext = file_path.split('.')[-1]
        new_name = os.path.join(folder_path, f"{idx}.{ext}")
        os.rename(file_path, new_name)
        print(f"Renamed: {file_path} -> {new_name}")

def run_inference_on_images(model_path, images_folder, save_folder="runs/detect"):
    """Run YOLO inference on all images in a folder."""
    model = YOLO(model_path)
    image_files = sorted(glob(os.path.join(images_folder, "*.jpg")))
    
    for img_path in image_files:
        print(f"Processing image: {img_path} ...")
        model(img_path, save=True)  # saved in runs/detect automatically
    
    print(f"Image inference done! Check {save_folder}.")

def run_inference_on_videos(model_path, videos_folder, save_folder="runs/detect"):
    """Run YOLO inference on all videos in a folder."""
    model = YOLO(model_path)
    video_files = sorted(glob(os.path.join(videos_folder, "*.mp4")))  # add more extensions if needed
    
    for video_path in video_files:
        print(f"Processing video: {video_path} ...")
        model(video_path, save=True)  # saved in runs/detect automatically
    
    print(f"Video inference done! Check {save_folder}.")

# Paths
images_folder = "inference/images"
videos_folder = "inference/videos"
model_path = "models/best.pt"

# Step 1: Rename files
rename_files(images_folder, ["jpg", "jpeg", "png"])
rename_files(videos_folder, ["mp4", "avi", "mov"])

# Step 2: Run YOLO on images
run_inference_on_images(model_path, images_folder)

# Step 3: Run YOLO on videos
run_inference_on_videos(model_path, videos_folder)
