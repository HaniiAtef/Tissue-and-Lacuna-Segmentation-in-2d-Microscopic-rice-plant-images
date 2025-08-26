import os
import cv2
import torch
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
from scipy.ndimage import label as cc_label
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------
# IMAGE ENHANCEMENT HELPERS
# ------------------------
def enhance_image(gray_image, clahe_clip=2.0, clahe_tile=(8, 8), gamma=1.0):
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    enhanced = clahe.apply(gray_image)
    if gamma != 1.0:
        lut = np.array([((i/255.0) ** (1.0/gamma)) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, lut)
    return enhanced

def overlay_mask(base_image, mask, color=(255,0,0), alpha=0.5):
    overlay = base_image.copy()
    color_mask = np.zeros_like(base_image)
    color_mask[mask==1] = color
    return cv2.addWeighted(color_mask, alpha, overlay, 1-alpha, 0)

def add_header(labels, panel_width=512, height=80, label_width=120):
    total_width = label_width + panel_width * len(labels)
    header = np.ones((height, total_width, 3), dtype=np.uint8) * 255
    for i, label in enumerate(labels):
        x = label_width + i * panel_width + 10
        cv2.putText(header, label, (x, int(height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2, cv2.LINE_AA)
    return header

def add_row_label(index, ar_percent, height, width=120):
    label_strip = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.putText(label_strip, f"{index}", (5, int(height*0.4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(label_strip, f"AR {ar_percent:.1f}%", (5, int(height*0.8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
    return label_strip

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]

def pad_slice_rows(rows, target_rows, panel_width=512, panel_height=341, label_width=120):
    blank_row = np.hstack([
        add_row_label(0, 0.0, height=panel_height, width=label_width),
        np.ones((panel_height, panel_width*3, 3), dtype=np.uint8)*255
    ])
    rows_padded = rows[:]
    while len(rows_padded)<target_rows:
        rows_padded.append(blank_row)
    return rows_padded

# ------------------------
# SEGFORMER PREDICTION CLASS
# ------------------------
class SegformerPredictor:
    def __init__(self, model_ar_path, model_ce_path, images_path, output_path):
        self.model_ar_path = model_ar_path
        self.model_ce_path = model_ce_path
        self.images_path = images_path
        self.output_path = output_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_size = (341, 512)

        # load models
        self.model_ar = SegformerForSemanticSegmentation.from_pretrained(self.model_ar_path).to(self.device).eval()
        self.model_ce = SegformerForSemanticSegmentation.from_pretrained(self.model_ce_path).to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1))  # grayscale to 3 channels
        ])

    def get_largest_component_mask(self, binary_mask):
        labeled_array, num_features = cc_label(binary_mask)
        if num_features==0:
            return binary_mask*0
        largest_component = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
        return (labeled_array==largest_component).astype(np.uint8)

    def predict_mask(self, model, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(image_tensor).logits
            upsampled = torch.nn.functional.interpolate(
                logits, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
            )
            return torch.argmax(upsampled, dim=1)[0].cpu().numpy()

    def create_mask_gray(self, pred, class_id):
        return np.where(pred==class_id, 255, 0).astype(np.uint8)

    def run_prediction(self):
        # prepare dirs
        pred_dir = os.path.join(self.output_path, 'predicted_images')
        ar_dir = os.path.join(pred_dir, 'pred_AR')
        cortex_dir = os.path.join(pred_dir, 'pred_Cortex')
        endo_dir = os.path.join(pred_dir, 'pred_Endoderm')
        raw_out_dir = os.path.join(self.output_path, 'raw_images')

        for d in [ar_dir, cortex_dir, endo_dir, raw_out_dir]:
            os.makedirs(d, exist_ok=True)

        rows = []
        for image_file in tqdm(os.listdir(self.images_path), desc="Predicting"):
            if not image_file.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
                continue
            img_path = os.path.join(self.images_path, image_file)
            image = Image.open(img_path).convert('L')

            # resize raw to 512x341 and enhance before saving
            raw_np = np.array(image)
            raw_resized = cv2.resize(raw_np, (512,341), interpolation=cv2.INTER_AREA)
            enhanced_raw = enhance_image(raw_resized, clahe_clip=2.0, clahe_tile=(8,8), gamma=1.2)
            cv2.imwrite(os.path.join(raw_out_dir, image_file), enhanced_raw)

            # predictions
            pred_ar = self.predict_mask(self.model_ar, image)
            pred_ce = self.predict_mask(self.model_ce, image)

            mask_ar = self.create_mask_gray(pred_ar,1)
            mask_cortex = self.create_mask_gray(pred_ce,1)
            mask_endo = self.create_mask_gray(pred_ce,2)

            # Save masks to separate folders
            tifffile.imwrite(os.path.join(ar_dir, image_file), mask_ar)
            tifffile.imwrite(os.path.join(cortex_dir, image_file), mask_cortex)
            tifffile.imwrite(os.path.join(endo_dir, image_file), mask_endo)

            # metric calculation (AR %)
            combined_ce = ((mask_cortex>0)|(mask_endo>0)).astype(np.uint8)
            largest_cc = self.get_largest_component_mask(combined_ce)
            ar_in_cc = ((mask_ar>0)&(largest_cc>0)).astype(np.uint8)
            cortex_in_cc = ((mask_cortex>0)&(largest_cc>0)).astype(np.uint8)

            total_cortex = cortex_in_cc.sum()
            ar_pixels = ar_in_cc.sum()
            rows.append({
                'image_name': image_file,
                'AR_percent_in_Cortex_only': (ar_pixels/total_cortex*100) if total_cortex>0 else 0
            })

        df_out = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_path,'pred_metrics.csv')
        df_out.to_csv(csv_path,index=False)
        print(f"Prediction complete. Raw copies and masks saved in {self.output_path}.")
        return ar_dir, cortex_dir, endo_dir, csv_path

# ------------------------
# VISUALIZATION FUNCTION (with distinct colors)
# ------------------------
def build_tiff(raw_dir, ar_dir, cortex_dir, endo_dir, csv_path, output_path, overlay_alpha=0.3):
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(by='AR_percent_in_Cortex_only', ascending=False).reset_index(drop=True)

    panel_rows = []
    idx = 1
    for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted), desc="Building panels"):
        fname = row['image_name']
        ar_percent = row['AR_percent_in_Cortex_only']
        raw_path = os.path.join(raw_dir, fname)
        ar_path = os.path.join(ar_dir, fname)
        cortex_path = os.path.join(cortex_dir, fname)
        endo_path = os.path.join(endo_dir, fname)

        if not all(os.path.exists(p) for p in [raw_path, ar_path, cortex_path, endo_path]):
            print(f"Skipping missing: {fname}")
            continue

        # Load images
        raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)

        # AR overlay (red)
        ar_mask = cv2.imread(ar_path, cv2.IMREAD_GRAYSCALE)
        ar_mask_bin = cv2.resize((ar_mask>127).astype(np.uint8), (512,341), interpolation=cv2.INTER_NEAREST)
        ar_overlay = overlay_mask(raw_img.copy(), ar_mask_bin, color=(255,0,0), alpha=overlay_alpha)

        # Cortex (green) and Endoderm (blue) overlay
        cortex_mask = cv2.imread(cortex_path, cv2.IMREAD_GRAYSCALE)
        endo_mask = cv2.imread(endo_path, cv2.IMREAD_GRAYSCALE)
        cortex_bin = cv2.resize((cortex_mask>127).astype(np.uint8), (512,341), interpolation=cv2.INTER_NEAREST)
        endo_bin = cv2.resize((endo_mask>127).astype(np.uint8), (512,341), interpolation=cv2.INTER_NEAREST)

        ce_overlay = raw_img.copy()
        ce_overlay = overlay_mask(ce_overlay, cortex_bin, color=(0,255,0), alpha=overlay_alpha)
        ce_overlay = overlay_mask(ce_overlay, endo_bin, color=(0,0,255), alpha=overlay_alpha)

        # Build row: AR % label | Raw | AR overlay | Cortex + Endoderm overlay
        row_img = np.hstack([
            add_row_label(idx, ar_percent, height=341),
            raw_img,
            ar_overlay,
            ce_overlay
        ])
        panel_rows.append(row_img)
        idx += 1

    if not panel_rows:
        print("No images processed.")
        return

    # Header for columns
    header = add_header(["Raw","AR","Cortex+Endoderm"], panel_width=512, height=80, label_width=120)
    slices = []
    for chunk in chunk_list(panel_rows, 3):
        padded = pad_slice_rows(chunk, 3, panel_width=512, panel_height=341, label_width=120)
        slice_img = np.vstack([header]+padded)
        slices.append(slice_img)

    final_stack = np.stack(slices, axis=0)
    tifffile.imwrite(output_path, final_stack, photometric='rgb')
    print(f"Visualization saved as 3D TIFF: {output_path}")

# ------------------------
# MAIN EXECUTION
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='/home/hmohamed/Documents/test_github/data/inference_images', help='Path to image folder')
    parser.add_argument('--output', type=str, default='/home/hmohamed/Documents/test_github/output_test', help='Directory to save masks')
    parser.add_argument('--model_ar', type=str, default='/home/hmohamed/Documents/test_github/Models/Lacuna_models/B2_model', help='Path to AR model')
    parser.add_argument('--model_ce', type=str, default='/home/hmohamed/Documents/test_github/Models/Cortex_Endo_models/B2_model', help='Path to CE model')
    parser.add_argument('--alpha', type=float, default=0.3, help='Overlay transparency (0.0 to 1.0)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    out_tif = os.path.join(args.output,"merged_output.tif")

    predictor = SegformerPredictor(args.model_ar, args.model_ce, args.images, args.output)
    ar_dir, cortex_dir, endo_dir, csv_path = predictor.run_prediction()
    build_tiff(os.path.join(args.output,'raw_images'), ar_dir, cortex_dir, endo_dir, csv_path, out_tif, overlay_alpha=args.alpha)

if __name__ == "__main__":
    main()
