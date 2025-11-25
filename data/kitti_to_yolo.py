# kitti_to_yolo.py

import os
import cv2

# Script Directory
script_dir = os.path.dirname(os.path.realpath(__file__))
# Parent Directory
main_dir = os.path.dirname(script_dir) # /home/amanu/Desktop/Projects/real_time_object_detection

# Configuration

IMG_DIR = f"{main_dir}/data/KITTI_Dataset/data_object_image/train"            # KITTI left camera images
LABEL_DIR = f"{main_dir}/data/KITTI_Dataset/data_object_label/labels"         # KITTI label folder
OUTPUT_LABEL_DIR = f"{main_dir}/data/KITTI_Dataset/data_object_label/labels_yolo"  # Output YOLO label folder
IMG_WIDTH = 1224                                                             # KITTI standard width
IMG_HEIGHT = 370                                                              # KITTI standard height

# Class mapping
CLASS_MAP = {
    "Car": 0,
    "Truck": 1,
    "Pedestrian": 2,
    "Cyclist": 3,
    "Van" : 4,
    "Person_sitting" : 5,
    "Tram" : 6,
    "Misc" : 7
}

os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def convert_bbox(x1, y1, x2, y2, img_width, img_height):
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return x_center, y_center, w, h

def process_labels():
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith(".txt")]
    for label_file in label_files:
        # Get corresponding image
        img_name = label_file.replace(".txt", ".png")
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}, skipping")
            continue
        img_height, img_width = img.shape[:2]

        output_lines = []
        with open(os.path.join(LABEL_DIR, label_file), "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                cls = parts[0]
                if cls not in CLASS_MAP:
                    continue  # skip DontCare and unknown
                x1, y1, x2, y2 = map(float, parts[4:8])
                x_c, y_c, w, h = convert_bbox(x1, y1, x2, y2, img_width, img_height)
                output_lines.append(f"{CLASS_MAP[cls]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        if output_lines:
            out_path = os.path.join(OUTPUT_LABEL_DIR, label_file)
            with open(out_path, "w") as out_f:
                out_f.write("\n".join(output_lines))

if __name__ == "__main__":
    process_labels()
    print(f"YOLO labels saved to {OUTPUT_LABEL_DIR}")
