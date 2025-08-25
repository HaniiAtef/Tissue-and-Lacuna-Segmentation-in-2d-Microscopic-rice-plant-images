import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
import tifffile
from scipy.ndimage import label
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class SegformerPredictor:
    def __init__(self, model_ar_path, model_ce_path, images_path, output_path):
        self.model_ar_path = model_ar_path
        self.model_ce_path = model_ce_path
        self.images_path = images_path
        self.output_path = output_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_size = (341, 512)

        # Load models
        self.model_ar = SegformerForSemanticSegmentation.from_pretrained(self.model_ar_path).to(self.device).eval()
        self.model_ce = SegformerForSemanticSegmentation.from_pretrained(self.model_ce_path).to(self.device).eval()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # convert grayscale to 3-channel
        ])

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

    def run_prediction(self):
        pred_dir = os.path.join(self.output_path, 'predicted_images')
        os.makedirs(pred_dir, exist_ok=True)
        ar_dir = os.path.join(pred_dir, 'pred_AR')
        cortex_dir = os.path.join(pred_dir, 'pred_Cortex')
        endo_dir = os.path.join(pred_dir, 'pred_Endoderm')
        os.makedirs(ar_dir, exist_ok=True)
        os.makedirs(cortex_dir, exist_ok=True)
        os.makedirs(endo_dir, exist_ok=True)

        rows = []
        for image_file in os.listdir(self.images_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue

            img_path = os.path.join(self.images_path, image_file)
            image = Image.open(img_path).convert('L')

            pred_ar = self.predict_mask(self.model_ar, image)
            pred_ce = self.predict_mask(self.model_ce, image)

            mask_ar = self.create_mask_gray(pred_ar, 1)
            mask_cortex = self.create_mask_gray(pred_ce, 1)
            mask_endo = self.create_mask_gray(pred_ce, 2)

            # Combine CE masks for largest component
            combined_ce = ((mask_cortex > 0) | (mask_endo > 0)).astype(np.uint8)
            largest_cc = self.get_largest_component_mask(combined_ce)

            # Crop AR within largest CE component
            ar_in_cc = ((mask_ar > 0) & (largest_cc > 0)).astype(np.uint8)
            cortex_in_cc = ((mask_cortex > 0) & (largest_cc > 0)).astype(np.uint8)

            total_ce_pixels = largest_cc.sum()
            total_cortex_pixels = cortex_in_cc.sum()
            ar_pixels = ar_in_cc.sum()

            metrics = {
                'image_name': image_file,
                'AR_percent_in_Cortex+Endoderm': (ar_pixels / total_ce_pixels * 100) if total_ce_pixels > 0 else 0,
                'AR_percent_in_Cortex_only': (ar_pixels / total_cortex_pixels * 100) if total_cortex_pixels > 0 else 0
            }
            rows.append(metrics)

            # Save masks
            tifffile.imwrite(os.path.join(ar_dir, image_file), mask_ar)
            tifffile.imwrite(os.path.join(cortex_dir, image_file), mask_cortex)
            tifffile.imwrite(os.path.join(endo_dir, image_file), mask_endo)

        # Save CSV
        df_out = pd.DataFrame(rows)
        df_out.to_csv(os.path.join(self.output_path, 'pred_metrics.csv'), index=False)
        print(f"Prediction complete. Masks and CSV saved in {self.output_path}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='/home/hmohamed/Documents/test_github/1_Source', help='Path to image or folder of images')
    parser.add_argument('--output', type=str, default='/home/hmohamed/Documents/test_github/output_test', help='Directory to save masks')
    parser.add_argument('--model_ar', type=str, default='/home/hmohamed/Documents/test_github/Models/Lacuna_models/B2_model', help='Path to AR model')
    parser.add_argument('--model_ce', type=str, default='/home/hmohamed/Documents/test_github/Models/Cortex_Endo_models/B2_model', help='Path to CE model')

    args = parser.parse_args()

    predictor = SegformerPredictor(args.model_ar, args.model_ce, args.images, args.output)
    predictor.run_prediction()


if __name__ == "__main__":
    main()
