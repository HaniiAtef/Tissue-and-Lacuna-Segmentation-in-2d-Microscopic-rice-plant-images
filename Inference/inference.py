import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
import tifffile
import cv2

class SegformerInference:
    def __init__(self, model_ar_path, model_ce_path, output_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_size = (341, 512)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load models
        self.model_ar = SegformerForSemanticSegmentation.from_pretrained(model_ar_path).to(self.device).eval()
        self.model_ce = SegformerForSemanticSegmentation.from_pretrained(model_ce_path).to(self.device).eval()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

    def apply_clahe(self, pil_image):
        img_np = np.array(pil_image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_np)
        return Image.fromarray(enhanced)

    def predict_mask(self, model, image):
        image = self.apply_clahe(image)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(image_tensor).logits
            upsampled = torch.nn.functional.interpolate(
                logits, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
            )
            return torch.argmax(upsampled, dim=1)[0].cpu().numpy()

    def create_mask_gray(self, pred, class_id):
        return np.where(pred == class_id, 255, 0).astype(np.uint8)

    def run_inference(self, image_paths):
        ar_dir = os.path.join(self.output_dir, 'pred_AR')
        cortex_dir = os.path.join(self.output_dir, 'pred_Cortex')
        endo_dir = os.path.join(self.output_dir, 'pred_Endoderm')
        os.makedirs(ar_dir, exist_ok=True)
        os.makedirs(cortex_dir, exist_ok=True)
        os.makedirs(endo_dir, exist_ok=True)

        print(f"Processing {len(image_paths)} images...")

        for img_path in image_paths:
            image_name = os.path.basename(img_path)
            image = Image.open(img_path).convert('L')

            # Predictions
            pred_ar = self.predict_mask(self.model_ar, image)
            pred_ce = self.predict_mask(self.model_ce, image)

            mask_ar = self.create_mask_gray(pred_ar, 1)
            mask_cortex = self.create_mask_gray(pred_ce, 1)
            mask_endo = self.create_mask_gray(pred_ce, 2)

            # Save masks
            tifffile.imwrite(os.path.join(ar_dir, image_name), mask_ar)
            tifffile.imwrite(os.path.join(cortex_dir, image_name), mask_cortex)
            tifffile.imwrite(os.path.join(endo_dir, image_name), mask_endo)
            print(f"Processed: {image_name}")

def get_image_paths(input_path):
    if not os.path.exists(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")

    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        if not files:
            raise ValueError(f"No image files found in folder: {input_path}")
        return sorted(files)

    elif os.path.isfile(input_path):
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            return [input_path]
        else:
            raise ValueError("Input file must be an image (png, jpg, jpeg, tif, tiff).")

    else:
        raise ValueError("Input path must be a folder or a single image file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='path/to/image_folder', help='Path to image or folder of images')
    parser.add_argument('--model_ar', type=str, default='../Models/Lacuna_models/B2_model', help='Path to AR model')
    parser.add_argument('--model_ce', type=str, default='../Models/Cortex_Endoderm_models/B2_model', help='Path to CE model')
    parser.add_argument('--output', type=str, default='../data/output_masks', help='Directory to save masks')
    args = parser.parse_args()

    image_paths = get_image_paths(args.input)
    inference = SegformerInference(args.model_ar, args.model_ce, args.output)
    inference.run_inference(image_paths)
