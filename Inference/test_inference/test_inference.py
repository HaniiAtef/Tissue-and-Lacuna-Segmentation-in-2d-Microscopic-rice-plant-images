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

warnings.filterwarnings("ignore", category=FutureWarning)

class SegformerDualMetrics:
    def __init__(self, model_ar_path, model_ce_path, root_path):
        self.model_ar_path = model_ar_path
        self.model_ce_path = model_ce_path
        self.root_path = root_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_size = (341, 512)
        # self.target_size = (681, 1024)
        # self.target_size = (1088, 1636)
        


        # Load models
        self.model_ar = SegformerForSemanticSegmentation.from_pretrained(self.model_ar_path).to(self.device).eval()
        self.model_ce = SegformerForSemanticSegmentation.from_pretrained(self.model_ce_path).to(self.device).eval()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

    def get_largest_component_mask(self, binary_mask):
        labeled_array, num_features = label(binary_mask)
        if num_features == 0:
            return binary_mask * 0  # All zeros if no component
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
                # Create subfolders
                ar_dir = os.path.join(pred_dir, 'pred_AR')
                cortex_dir = os.path.join(pred_dir, 'pred_Cortex')
                endo_dir = os.path.join(pred_dir, 'pred_Endoderm')
                os.makedirs(ar_dir, exist_ok=True)
                os.makedirs(cortex_dir, exist_ok=True)
                os.makedirs(endo_dir, exist_ok=True)

                # Save individual masks
                tifffile.imwrite(os.path.join(ar_dir, image_file), mask_ar)
                tifffile.imwrite(os.path.join(cortex_dir, image_file), mask_cortex)
                tifffile.imwrite(os.path.join(endo_dir, image_file), mask_endo)


            df_out = pd.DataFrame(rows)
            df_out.to_csv(os.path.join(subfolder_path, 'pred_metrics.csv'), index=False)

    def metric_comparison(self):
        for subfolder in sorted(f for f in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, f))):
            path = os.path.join(self.root_path, subfolder)
            try:
                real_df = pd.read_csv(os.path.join(path, 'output.csv'))
                pred_df = pd.read_csv(os.path.join(path, 'pred_metrics.csv'))
                merged = pd.merge(real_df, pred_df, left_on='0_PARAM_ImgName', right_on='image_name')

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uc', type=str, default='path/to/use_case', help='Root directory path')
    parser.add_argument('--model_ar', type=str, default='../../Models/Lacuna_models/B2_model', help='Path to AR model')
    parser.add_argument('--model_ce', type=str, default='../../Models/Cortex_Endoderm_models/B2_model', help='Path to CE model')
    args = parser.parse_args()

    metrics = SegformerDualMetrics(args.model_ar, args.model_ce, args.uc)
    metrics.metrics_generation()
    metrics.metric_comparison()

if __name__ == "__main__":
    main()
