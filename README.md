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
│   └── B2_model/         # AR model
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

## Citation

If you use this repository, please cite the original work or link back to this repo.

---

## License

This project is released under the **MIT License**.  
See [LICENSE](LICENSE) for details.
