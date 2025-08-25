import os
import sys
import glob
import cv2
import numpy as np
from tqdm import tqdm

# Configuration
KERNEL_SIZE = 3
ITERATIONS = 1
MASK_FOLDERS = ['AR_mask', 'Cortex_mask', 'Endoderm_mask']
AUGMENTATION_TYPES = ['rotate', 'flip', 'translate']

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Unable to load image: {path}")
    return img

def save_image(img, output_path):
    if not cv2.imwrite(output_path, img):
        raise ValueError(f"❌ Failed to save image: {output_path}")

def rotate_image(img, angle):
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def flip_image(img, flip_code):
    return cv2.flip(img, flip_code)

def translate_image(img, tx, ty):
    h, w = img.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def apply_morphological_closing(img):
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=ITERATIONS)
    return closed

def process_image_folder(image_folder):
    """Process original images with augmentations"""
    print(f"\n🔧 Processing original images in: {image_folder}")
    
    for img_path in tqdm(glob.glob(os.path.join(image_folder, "*.tif")), desc="Augmenting images"):
        try:
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            img = load_image(img_path)

            # Rotations every 15 degrees (15, 30, ..., 345)
            for angle in range(15, 360, 15):
                rotated = rotate_image(img, angle)
                output_path = os.path.join(image_folder, f"{name}_{angle}deg{ext}")
                save_image(rotated, output_path)

            # Flips: horizontal, vertical, both
            flips = [("flipH", 1), ("flipV", 0), ("flipHV", -1)]
            for suffix, flip_code in flips:
                flipped = flip_image(img, flip_code)
                output_path = os.path.join(image_folder, f"{name}_{suffix}{ext}")
                save_image(flipped, output_path)

            # Translations: ±10 pixels in x and y
            translations = [
                ("transXpos", 10, 0),
                ("transXneg", -10, 0),
                ("transYpos", 0, 10),
                ("transYneg", 0, -10)
            ]
            for suffix, tx, ty in translations:
                translated = translate_image(img, tx, ty)
                output_path = os.path.join(image_folder, f"{name}_{suffix}{ext}")
                save_image(translated, output_path)

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

def process_mask_folder(mask_folder):
    """Process mask folder with closing and matching augmentations"""
    print(f"\n🔧 Processing mask folder: {mask_folder}")
    
    for mask_path in tqdm(glob.glob(os.path.join(mask_folder, "*.tif")), desc="Processing masks"):
        try:
            filename = os.path.basename(mask_path)
            name, ext = os.path.splitext(filename)
            mask = load_image(mask_path)
            
            # Apply morphological closing
            closed_mask = apply_morphological_closing(mask)
            save_image(closed_mask, mask_path)  # Overwrite original

            # Apply same augmentations as original images
            for angle in range(15, 360, 15):
                rotated = rotate_image(closed_mask, angle)
                output_path = os.path.join(mask_folder, f"{name}_{angle}deg{ext}")
                save_image(rotated, output_path)

            flips = [("flipH", 1), ("flipV", 0), ("flipHV", -1)]
            for suffix, flip_code in flips:
                flipped = flip_image(closed_mask, flip_code)
                output_path = os.path.join(mask_folder, f"{name}_{suffix}{ext}")
                save_image(flipped, output_path)

            translations = [
                ("transXpos", 10, 0),
                ("transXneg", -10, 0),
                ("transYpos", 0, 10),
                ("transYneg", 0, -10)
            ]
            for suffix, tx, ty in translations:
                translated = translate_image(closed_mask, tx, ty)
                output_path = os.path.join(mask_folder, f"{name}_{suffix}{ext}")
                save_image(translated, output_path)

        except Exception as e:
            print(f"❌ Error processing {mask_path}: {e}")

def process_experiment_folder(exp_folder):
    """Process one experiment folder"""
    print(f"\n🚀 Processing experiment: {exp_folder}")
    
    # Process original images
    original_images_folder = os.path.join(exp_folder, "Original_images")
    if os.path.exists(original_images_folder):
        process_image_folder(original_images_folder)
    else:
        print(f"⚠️ Original images folder not found: {original_images_folder}")

    # Process all mask folders
    for mask_folder_name in MASK_FOLDERS:
        mask_folder_path = os.path.join(exp_folder, mask_folder_name)
        if os.path.exists(mask_folder_path):
            process_mask_folder(mask_folder_path)
        else:
            print(f"⚠️ Mask folder not found: {mask_folder_path}")

def main():
    if len(sys.argv) < 2:
        print("❌ Please provide the project root path as an argument.")
        print("Usage: python3 augment_data.py /path/to/project/root")
        sys.exit(1)

    project_root = sys.argv[1]
    if not os.path.isdir(project_root):
        print(f"❌ Error: {project_root} is not a valid directory.")
        sys.exit(1)

    print(f"🚀 Starting data augmentation for: {project_root}")

    # Find all experiment folders
    experiment_folders = [
        os.path.join(project_root, d)
        for d in os.listdir(project_root)
        if os.path.isdir(os.path.join(project_root, d))
    ]

    for exp_folder in experiment_folders:
        process_experiment_folder(exp_folder)

    print("✅ All augmentations completed!")

if __name__ == "__main__":
    main()