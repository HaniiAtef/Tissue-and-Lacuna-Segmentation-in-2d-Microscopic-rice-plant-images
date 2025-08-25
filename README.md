# Tissue-and-Lacuna-Segmentation-in-2d-microscopic-images-of-rice-plant

This repository provides training and inference code for segmenting **Tissue Cortex/Endoderm (CE)** and **Lacuna (AR)** regions in 2D microscopic rice plant images using [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) models.

---

## Repository Structure

```
.
├── Train/
│   ├── dataset.py                  # Data loading and preprocessing
│   ├── train.py                    # Training script for SegFormer models
│
└── Inference/
    ├── inference.py                # Main inference script
    ├── test_inference/
    │   ├── Pred_vs_true_ar.py       # Compare predicted AR masks with ground truth
    │   ├── test_CLAHE_filter_inference.py  # Test CLAHE preprocessing with inference
    │   └── test_inference.py        # Quick inference test
```

Pretrained models (6 total: 3 models for the Lacuna and 3 models for Cortex and Endo) are available via Google Drive:  
**[Download models here](https://drive.google.com/drive/folders/1Xd785QHcLC2CnkLLYYvWH1R51p2i2uLS?usp=drive_link)**

---

## 1. Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/HaniiAtef/Tissue-and-Lacuna-Segmentation-in-2d-Microscopic-rice-plant-images.git

```
```bash
cd Tissue-and-Lacuna-Segmentation-in-2d-Microscopic-rice-plant-images
```

# (Recommended) Create a virtual environment
```bash
python3 -m venv venv
```
```bash
source venv/bin/activate   # Linux / MacOS
```

```bash
venv\Scripts\activate      # Windows
```


# Install required packages
```bash
pip install -r requirements.txt
```

> **Note:** Ensure you have **Python 3.8+** and **torch** installed with CUDA if you plan to use GPU acceleration.

---

## 2. Download Pretrained Models

Download the **three SegFormer models** from Google Drive and place them in:

```
Models/
├── Lacuna_models/
    └── B2_model/         # AR model
    └── B3_model/
    └── B4_model/ 
└── Cortex_Endo_models/
    └── B2_model/         # CE model
    └── B3_model/
    └── B4_model/ 
```

---

## 3. Training (Optional)

If you wish to train your own models instead of using the pretrained ones:

```bash
cd Train
python train.py     --dataset_path /path/to/dataset     --output_dir /path/to/save/model     --epochs 50     --batch_size 8
```

Adjust dataset paths, epochs, and hyperparameters as needed.  
The `dataset.py` handles loading and preprocessing automatically.


## Training Dataset Structure

The `CellDataset` class expects a **folder hierarchy** and naming convention like this:


```
All_Use_Cases/
├── UC1/
│ ├── Experiment1/
│ │ ├── Original_images/ # Input images
│ │ │ ├── img1.tif
│ │ │ ├── img2.tif
│ │ │ └── ...
│ │ ├── AR_mask/ # Lacuna masks
│ │ │ ├── img1_image_mask_ar.tif
│ │ │ └── ...
│ │ ├── Cortex_mask/ # Cortex masks
│ │ │ ├── img1_image_mask_cortex.tif
│ │ │ └── ...
│ │ └── Endoderm_mask/ # Endoderm masks
│ │ ├── img1_image_mask_endoderm.tif
│ │ └── ...
│ ├── Experiment2/
│ │ └── ... (same structure)
├── UC2/
│ └── ... (same structure)
└── ...

```

### Key Points

1. **Root folder**: `All_Use_Cases/` contains all use cases (`UC1`, `UC2`, …).  
2. **Use Cases (UC)**: Each `UC` folder contains multiple experiments (e.g., `Experiment1`, `Experiment2`).  
3. **Original Images**: Stored in `Original_images/` inside each experiment. Must be `.tif` files.  
4. **Masks**:  
   - `AR_mask/` → Lacuna mask  
   - `Cortex_mask/` → Cortex mask  
   - `Endoderm_mask/` → Endoderm mask  
5. **Mask Filenames**: Must match the image filename **with a suffix**:  
   - AR mask → `_image_mask_ar`  
   - Cortex mask → `_image_mask_cortex`  
   - Endoderm mask → `_image_mask_endoderm`  

   Example: `img1.tif` → AR mask: `img1_image_mask_ar.tif`  
6. **Mask Types**:  
   - Single mask type → Binary mode  
   - Multiple mask types → Multi-class mode; labels assigned automatically in order of `mask_types` (1, 2, 3…)  
7. **Split**: Each experiment folder is automatically split into:  
   - **Train**: first 85% of images  
   - **Validation**: last 15% of images  
8. **Transforms**:  
   - Images resized to `(341, 512)` and converted to 3-channel tensors  
   - Masks resized using nearest neighbor interpolation  
   - Multi-class masks combined into a single labeled mask  
   - Binary masks converted to 0/1 tensors

---

## 4. Running Inference

The main inference script is `Inference/inference.py`.

```bash
cd Inference

python inference.py     --images /path/to/images     --output /path/to/output_dir     --model_ar /path/to/Lacuna_models/B2_model     --model_ce /path/to/Cortex_Endo_models/B2_model
```

### What it does:
- Loads the pretrained **AR** and **CE** models  
- Processes each image (supports `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`)  
- Generates **AR masks**, **Cortex masks**, **Endoderm masks**  
- Saves them into `predicted_images/` subfolders  
- Outputs a CSV `pred_metrics.csv` containing:  
  - `% AR inside Cortex+Endoderm`  
  - `% AR inside Cortex only`  

---

## 5. Testing and Visualization

Several helper scripts are in `Inference/test_inference/`:

- `Pred_vs_true_ar.py` – Compare predicted AR masks with ground truth  
- `test_CLAHE_filter_inference.py` – Test CLAHE preprocessing with inference  
- `test_inference.py` – Quick run for debugging or verifying inference works  

Run them directly, e.g.:

```bash
cd Inference/test_inference
python test_inference.py --images /path/to/images --model_ar /path/to/model
```

---

## 6. Example Output

- **Masks:**  
  - `predicted_images/pred_AR/` – AR segmentation  
  - `predicted_images/pred_Cortex/` – Cortex segmentation  
  - `predicted_images/pred_Endoderm/` – Endoderm segmentation  

- **Metrics:**  
  CSV file `pred_metrics.csv` contains AR coverage percentages per image.

---


