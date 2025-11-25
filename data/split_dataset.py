import os
import random
import shutil


# Script Directory
script_dir = os.path.dirname(os.path.realpath(__file__))
# Parent Directory
main_dir = os.path.dirname(script_dir) # /home/amanu/Desktop/Projects/Real-Time Object detection
# Paths configuration

IMAGE_DIR = f"{main_dir}/data/KITTI_Dataset/data_object_image/train"
LABEL_DIR = f"{main_dir}/data/KITTI_Dataset/data_object_label/labels_yolo"

OUTPUT_DIR = f"{main_dir}/final_dataset"
TRAIN_RATIO = 0.8  # 80% train, 20% val

# Create output folders

train_img_out = os.path.join(OUTPUT_DIR, "train/images")
train_lbl_out = os.path.join(OUTPUT_DIR, "train/labels")
val_img_out = os.path.join(OUTPUT_DIR, "val/images")
val_lbl_out = os.path.join(OUTPUT_DIR, "val/labels")

for folder in [train_img_out, train_lbl_out, val_img_out, val_lbl_out]:
    os.makedirs(folder, exist_ok=True)

# Load image files

images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
images.sort()

print(f"Found {len(images)} images in train/image_2")


# Shuffle and Split

random.shuffle(images)
split_index = int(len(images) * TRAIN_RATIO)
train_files = images[:split_index]
val_files   = images[split_index:]

print(f"→ Train: {len(train_files)} images")
print(f"→ Val:   {len(val_files)} images")


# Copy pairs of labels

def copy_image_and_label(filename, dest_img, dest_lbl):
    img_src = os.path.join(IMAGE_DIR, filename)
    
    # Label must have same filename but .txt
    label_name = os.path.splitext(filename)[0] + ".txt"
    lbl_src = os.path.join(LABEL_DIR, label_name)

    img_dest = os.path.join(dest_img, filename)
    lbl_dest = os.path.join(dest_lbl, label_name)

    shutil.copy(img_src, img_dest)

    if os.path.exists(lbl_src):
        shutil.copy(lbl_src, lbl_dest)
    else:
        # Create an empty label file if missing
        open(lbl_dest, "w").close()

# Copy train data

print("Copying train files...")
for f in train_files:
    copy_image_and_label(f, train_img_out, train_lbl_out)

# Copy val data

print("Copying val files...")
for f in val_files:
    copy_image_and_label(f, val_img_out, val_lbl_out)

print("✔ Dataset split completed successfully!")
