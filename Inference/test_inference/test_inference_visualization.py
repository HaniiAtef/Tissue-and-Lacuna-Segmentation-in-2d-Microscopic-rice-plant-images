import os
import fnmatch
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
import tifffile
import argparse
import warnings
import csv
from scipy.ndimage import label
import cv2

warnings.filterwarnings("ignore", category=FutureWarning)

class SegformerDualMetrics:
    def __init__(self, model_ar_path, model_ce_path, root_path):
        self.model_ar_path = model_ar_path
        self.model_ce_path = model_ce_path
        self.root_path = root_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_size = (341, 512)  # height, width

        # Load models
        self.model_ar = SegformerForSemanticSegmentation.from_pretrained(self.model_ar_path).to(self.device).eval()
        self.model_ce = SegformerForSemanticSegmentation.from_pretrained(self.model_ce_path).to(self.device).eval()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

    # ----------------------
    # MASK & METRICS HELPERS
    # ----------------------
    def get_largest_component_mask(self, binary_mask):
        labeled_array, num_features = label(binary_mask)
        if num_features == 0:
            return binary_mask * 0
        largest_component = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
        return (labeled_array == largest_component).astype(np.uint8)

    def predict_mask(self, model, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(image_tensor).logits
            upsampled = torch.nn.functional.interpolate(
                logits, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
            )
            return torch.argmax(upsampled, dim=1)[0].cpu().numpy()

    def create_mask_gray(self, pred, class_id):
        return np.where(pred == class_id, 255, 0).astype(np.uint8)

    # ----------------------
    # METRICS GENERATION
    # ----------------------
    def metrics_generation(self):
        for subfolder in sorted(f for f in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, f))):
            subfolder_path = os.path.join(self.root_path, subfolder)
            pred_dir = os.path.join(subfolder_path, 'predicted_images')
            os.makedirs(pred_dir, exist_ok=True)

            # Process CSV to extract ground truth ratios
            for csv_file in fnmatch.filter(os.listdir(subfolder_path), '*csv'):
                if csv_file == 'perimeters.csv':
                    continue
                csv_path = os.path.join(subfolder_path, csv_file)
                with open(csv_path, 'r') as f:
                    sample = f.read(2048)
                    try:
                        delimiter = csv.Sniffer().sniff(sample).delimiter
                    except csv.Error:
                        delimiter = ','

                try:
                    df = pd.read_csv(csv_path, sep=delimiter)
                except pd.errors.ParserError:
                    df = pd.read_csv(csv_path, sep=';')

                if 'Column1' in df.columns:
                    df.columns = df.iloc[0]
                    df = df[1:]
                    df.reset_index(drop=True, inplace=True)

                try:
                    s_df = df[['0_PARAM_ImgName', '12_COMPUTED_lacune_ratio_percent']]
                    s_df.to_csv(os.path.join(subfolder_path, 'output.csv'), index=False)
                except KeyError:
                    continue

            # Prediction and metric extraction
            rows = []
            images_path = os.path.join(subfolder_path, '1_Source')
            for image_file in os.listdir(images_path):
                img_path = os.path.join(images_path, image_file)
                image = Image.open(img_path).convert('L')

                pred_ar = self.predict_mask(self.model_ar, image)
                pred_ce = self.predict_mask(self.model_ce, image)

                mask_ar = self.create_mask_gray(pred_ar, 1)
                mask_cortex = self.create_mask_gray(pred_ce, 1)
                mask_endo = self.create_mask_gray(pred_ce, 2)

                # Combine CE masks for largest CC
                combined_ce = ((mask_cortex > 0) | (mask_endo > 0)).astype(np.uint8)
                largest_cc = self.get_largest_component_mask(combined_ce)

                # Crop AR within largest CE component
                ar_in_cc = ((mask_ar > 0) & (largest_cc > 0)).astype(np.uint8)
                cortex_in_cc = ((mask_cortex > 0) & (largest_cc > 0)).astype(np.uint8)

                # Metrics
                total_ce_pixels = largest_cc.sum()
                total_cortex_pixels = cortex_in_cc.sum()
                ar_pixels = ar_in_cc.sum()

                metrics = {
                    'image_name': image_file,
                    'AR_percent_in_Cortex+Endoderm': (ar_pixels / total_ce_pixels * 100) if total_ce_pixels > 0 else 0,
                    'AR_percent_in_Cortex_only': (ar_pixels / total_cortex_pixels * 100) if total_cortex_pixels > 0 else 0
                }
                rows.append(metrics)

                # Save predicted masks
                ar_dir = os.path.join(pred_dir, 'pred_AR')
                cortex_dir = os.path.join(pred_dir, 'pred_Cortex')
                endo_dir = os.path.join(pred_dir, 'pred_Endoderm')
                raw_dir = os.path.join(pred_dir, 'raw_images')
                for d in [ar_dir, cortex_dir, endo_dir, raw_dir]:
                    os.makedirs(d, exist_ok=True)

                tifffile.imwrite(os.path.join(ar_dir, image_file), mask_ar)
                tifffile.imwrite(os.path.join(cortex_dir, image_file), mask_cortex)
                tifffile.imwrite(os.path.join(endo_dir, image_file), mask_endo)

                # Save raw grayscale resized to 512x341
                raw_np = np.array(image.resize((512, 341)))
                cv2.imwrite(os.path.join(raw_dir, image_file), raw_np)

            df_out = pd.DataFrame(rows)
            metrics_csv = os.path.join(subfolder_path, 'pred_metrics.csv')
            df_out.to_csv(metrics_csv, index=False)

            # Build merged 3D TIFF visualization
            self.build_merged_tiff(raw_dir, ar_dir, os.path.join(pred_dir, 'merged_output.tif'),
                                   metrics_csv=metrics_csv)

    # ----------------------
    # METRIC COMPARISON
    # ----------------------
    def metric_comparison(self):
        for subfolder in sorted(f for f in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, f))):
            path = os.path.join(self.root_path, subfolder)
            try:
                real_df = pd.read_csv(os.path.join(path, 'output.csv'))
                pred_df = pd.read_csv(os.path.join(path, 'pred_metrics.csv'))
                merged = pd.merge(real_df, pred_df, left_on='0_PARAM_ImgName', right_on='image_name')
                merged.drop(columns=['image_name'], inplace=True)

                merged['diff_AR_in_CE'] = merged['12_COMPUTED_lacune_ratio_percent'] - merged['AR_percent_in_Cortex+Endoderm']
                merged['diff_AR_in_Cortex'] = merged['12_COMPUTED_lacune_ratio_percent'] - merged['AR_percent_in_Cortex_only']

                merged['abs_diff_AR_in_CE'] = merged['diff_AR_in_CE'].abs()
                merged['abs_diff_AR_in_Cortex'] = merged['diff_AR_in_Cortex'].abs()

                mean_df = pd.DataFrame({
                    '0_PARAM_ImgName': ['Mean'],
                    '12_COMPUTED_lacune_ratio_percent': [None],
                    'AR_percent_in_Cortex+Endoderm': [None],
                    'AR_percent_in_Cortex_only': [None],
                    'diff_AR_in_CE': [None],
                    'diff_AR_in_Cortex': [None],
                    'abs_diff_AR_in_CE': [merged['abs_diff_AR_in_CE'].mean()],
                    'abs_diff_AR_in_Cortex': [merged['abs_diff_AR_in_Cortex'].mean()]
                })

                merged = pd.concat([merged, mean_df], ignore_index=True)
                merged.to_csv(os.path.join(path, 'comparison_metrics.csv'), index=False)

            except Exception as e:
                print(f"Skipping {subfolder} due to error: {e}")

    # ----------------------
    # MERGED 3D TIFF FUNCTION
    # ----------------------
    def build_merged_tiff(self, raw_dir, ar_dir, output_path, alpha=0.3, metrics_csv=None):
        if metrics_csv is None:
            raise ValueError("metrics_csv path must be provided")
        df = pd.read_csv(metrics_csv)

        def overlay_mask(base_image, mask, color=(255,0,0), alpha=0.3):
            overlay = base_image.copy()
            color_mask = np.zeros_like(base_image)
            color_mask[mask>0] = color
            return cv2.addWeighted(color_mask, alpha, overlay, 1-alpha, 0)

        def add_header(labels, panel_width=512, height=80, label_width=120):
            total_width = label_width + panel_width * len(labels)
            header = np.ones((height, total_width, 3), dtype=np.uint8)*255
            for i, label in enumerate(labels):
                x = label_width + i*panel_width + 10
                cv2.putText(header, label, (x,int(height*0.7)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0),2)
            return header

        def add_row_label(idx, ar_percent, height, width=120):
            label_strip = np.ones((height,width,3),dtype=np.uint8)*255
            cv2.putText(label_strip, f"{idx}", (5,int(height*0.4)), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2)
            cv2.putText(label_strip, f"AR {ar_percent:.1f}%", (5,int(height*0.8)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
            return label_strip

        df_sorted = df.sort_values(by='AR_percent_in_Cortex_only', ascending=False).reset_index(drop=True)

        panel_rows=[]
        idx=1
        for _, row in df_sorted.iterrows():
            fname = row['image_name']
            ar_percent = row['AR_percent_in_Cortex_only']
            raw_path = os.path.join(raw_dir, fname)
            ar_mask_path = os.path.join(ar_dir, fname)
            if not os.path.exists(raw_path) or not os.path.exists(ar_mask_path):
                continue

            raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
            base_rgb = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)

            mask = cv2.imread(ar_mask_path, cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask, (512, 341), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_resized > 127).astype(np.uint8)
            overlay = overlay_mask(base_rgb, mask_bin, color=(255,0,0), alpha=alpha)

            row_img = np.hstack([add_row_label(idx, ar_percent, height=raw_img.shape[0]), base_rgb, overlay])
            panel_rows.append(row_img)
            idx+=1

        if not panel_rows:
            print("No images to build TIFF.")
            return

        header = add_header(["Raw","Predicted (AR)"], panel_width=512, height=80, label_width=120)
        final_stack = np.stack([np.vstack([header]+panel_rows)], axis=0)
        tifffile.imwrite(output_path, final_stack, photometric='rgb')
        print(f"Merged visualization saved at {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uc', type=str, default='/home/hmohamed/Documents/test_github/data/UC_test_data/UC3-DOMI-VARI', help='Root directory path')
    parser.add_argument('--model_ar', type=str, default='/home/hmohamed/Documents/test_github/Models/Lacuna_models/B2_model', help='Path to AR model')
    parser.add_argument('--model_ce', type=str, default='/home/hmohamed/Documents/test_github/Models/Cortex_Endo_models/B2_model', help='Path to CE model')
    args = parser.parse_args()

    metrics = SegformerDualMetrics(args.model_ar, args.model_ce, args.uc)
    metrics.metrics_generation()
    metrics.metric_comparison()


if __name__ == "__main__":
    main()
