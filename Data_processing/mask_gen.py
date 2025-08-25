import os
import sys
import glob
import zipfile
import roifile
import numpy as np
import tifffile
import pandas as pd
import shutil
from skimage.draw import polygon
import imagecodecs

def extract_roi_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def create_polygon_mask(image_shape, coordinates):
    mask = np.zeros(image_shape, dtype=np.uint8)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    rr, cc = polygon(y, x, shape=image_shape)
    mask[rr, cc] = 255
    return mask

def convert_to_grayscale(image):
    if image.ndim == 3:
        return np.mean(image, axis=-1).astype(np.uint8)
    return image.astype(np.uint8)

def save_tiff_with_check(image, output_path):
    image = convert_to_grayscale(image)
    image = np.ascontiguousarray(image)
    tifffile.imwrite(
        output_path,
        image,
        dtype=np.uint8,
        compression='zlib',
        photometric='minisblack'
    )
    print(f"Saved mask: {output_path}")

def reindex_roi_files(roi_folder):
    roi_files = sorted(glob.glob(os.path.join(roi_folder, '*.roi')))
    renamed_files = {}
    for new_index, roi_path in enumerate(roi_files, start=1):
        old_name = os.path.basename(roi_path)
        suffix = old_name[4:]
        new_name = f"{new_index:04d}-{suffix}"
        new_path = os.path.join(roi_folder, new_name)
        os.rename(roi_path, new_path)
        renamed_files[new_index] = new_path
    return renamed_files

def process_area_masks(image_path, roi_zip_paths, output_folders, naming_prefix):
    temp_folders = {
        roi_type: os.path.join(os.path.dirname(output_folders['cortex_in']), f'temp_{roi_type}')
        for roi_type in roi_zip_paths
    }

    for temp_folder in temp_folders.values():
        os.makedirs(temp_folder, exist_ok=True)

    for roi_type, zip_path in roi_zip_paths.items():
        if os.path.exists(zip_path):
            extract_roi_zip(zip_path, temp_folders[roi_type])
        else:
            print(f"ROI zip not found: {zip_path}")
            continue

    original_image = tifffile.imread(image_path)
    original_image = convert_to_grayscale(original_image)
    image_real_name = os.path.basename(image_path)

    original_masks = {}
    for roi_type, temp_folder in temp_folders.items():
        if not os.path.exists(roi_zip_paths[roi_type]):
            continue

        binary_mask = np.zeros(original_image.shape, dtype=np.uint8)
        roi_files = glob.glob(os.path.join(temp_folder, '*.roi'))

        for roi_path in roi_files:
            try:
                roi = roifile.roiread(roi_path)
                coords = roi.coordinates()
                roi_mask = create_polygon_mask(original_image.shape, coords)
                binary_mask = np.maximum(binary_mask, roi_mask)
            except Exception as e:
                print(f"Error reading ROI from {roi_path}: {e}")

        original_masks[roi_type] = binary_mask

    if 'cortex_in' in original_masks and 'stele_out' in original_masks:
        endoderm_mask = original_masks['cortex_in'].copy()
        stele_mask = original_masks['stele_out']
        combined_endoderm = np.maximum(endoderm_mask, stele_mask)

        output_path = os.path.join(output_folders['cortex_in'],
                                   f"{naming_prefix}{image_real_name}_image_mask_endoderm.tif")
        save_tiff_with_check(combined_endoderm, output_path)

        if 'cortex_convexhull' in original_masks:
            cortex_mask = original_masks['cortex_convexhull'].copy()
            cortex_mask[combined_endoderm == 255] = 0

            output_path = os.path.join(output_folders['cortex_convexhull'],
                                       f"{naming_prefix}{image_real_name}_image_mask_cortex.tif")
            save_tiff_with_check(cortex_mask, output_path)

    for temp_folder in temp_folders.values():
        shutil.rmtree(temp_folder, ignore_errors=True)

def process_image_rois(image_path, csv_path, roi_folder, output_folder, naming_prefix):
    os.makedirs(output_folder, exist_ok=True)
    image_real_name = os.path.basename(image_path)

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        df = None

    if df is None or df.empty:
        binary_mask = np.zeros(convert_to_grayscale(tifffile.imread(image_path)).shape, dtype=np.uint8)
        output_filename = f"{naming_prefix}{image_real_name}_image_mask_ar.tif"
        output_path = os.path.join(output_folder, output_filename)
        save_tiff_with_check(binary_mask, output_path)
        print(f"Empty AR mask saved for: {image_real_name}")
        return

    if 'ImgNameLac' in df.columns:
        roi_column = 'ImgNameLac'
        invert_selection = False
    elif 'ImgName' in df.columns:
        roi_column = 'ImgName'
        invert_selection = False
    elif 'ImgNameNoLac' in df.columns:
        roi_column = 'ImgNameNoLac'
        invert_selection = True
    else:
        print(f"No valid ROI column in {csv_path}")
        return

    roi_list_csv = df[df[roi_column] == image_real_name]['Displayed index (1-inf)'].tolist()
    all_rois_dict = reindex_roi_files(roi_folder)

    roi_list = (
        [i for i in all_rois_dict if i not in roi_list_csv]
        if invert_selection else
        [i for i in roi_list_csv if i in all_rois_dict]
    )

    original_image = tifffile.imread(image_path)
    original_image = convert_to_grayscale(original_image)
    binary_mask = np.zeros(original_image.shape, dtype=np.uint8)

    for roi_number in roi_list:
        roi_path = all_rois_dict.get(roi_number)
        if roi_path:
            try:
                roi = roifile.roiread(roi_path)
                coords = roi.coordinates()
                roi_mask = create_polygon_mask(original_image.shape, coords)
                binary_mask = np.maximum(binary_mask, roi_mask)
            except Exception as e:
                print(f"Error reading ROI {roi_number} from {roi_path}: {e}")

    output_filename = f"{naming_prefix}{image_real_name}_image_mask_ar.tif"
    output_path = os.path.join(output_folder, output_filename)
    save_tiff_with_check(binary_mask, output_path)

def process_experiment(experiment_root, project_prefix):
    source_folder = os.path.join(experiment_root, '1_Source')
    area_roi_folder = os.path.join(experiment_root, '2_AreaRoi')
    cell_roi_folder = os.path.join(experiment_root, '3_CellRoi')
    lacuna_indices_folder = os.path.join(experiment_root, '4_LacunesIndices')

    area_mask_folders = {
        'cortex_convexhull': os.path.join(experiment_root, 'Cortex_mask'),
        'cortex_in': os.path.join(experiment_root, 'Endoderm_mask')
    }
    ar_mask_folder = os.path.join(experiment_root, 'AR_mask')
    original_images_folder = os.path.join(experiment_root, 'Original_images')

    for folder in list(area_mask_folders.values()) + [ar_mask_folder, original_images_folder]:
        os.makedirs(folder, exist_ok=True)

    source_images = glob.glob(os.path.join(source_folder, '*.tif'))
    naming_prefix = f"{project_prefix}_{os.path.basename(experiment_root)}_"

    for img_path in source_images:
        original_name = os.path.basename(img_path)
        new_name = f"{naming_prefix}{original_name}"
        output_path = os.path.join(original_images_folder, new_name)
        shutil.copy(img_path, output_path)
        print(f"Copied original image as: {output_path}")

    temp_roi_folder = os.path.join(experiment_root, 'temp_roi_extraction')
    os.makedirs(temp_roi_folder, exist_ok=True)

    roi_zip_files = glob.glob(os.path.join(cell_roi_folder, '*.tif.zip'))

    for roi_zip_path in roi_zip_files:
        for f in os.listdir(temp_roi_folder):
            os.remove(os.path.join(temp_roi_folder, f))

        extract_roi_zip(roi_zip_path, temp_roi_folder)

        image_name = os.path.splitext(os.path.basename(roi_zip_path))[0]
        image_path = os.path.join(source_folder, image_name)
        csv_path = os.path.join(lacuna_indices_folder, f'{image_name}.csv')

        if os.path.exists(image_path) and os.path.exists(csv_path):
            process_image_rois(image_path, csv_path, temp_roi_folder, ar_mask_folder, naming_prefix)

    shutil.rmtree(temp_roi_folder, ignore_errors=True)

    for image_path in source_images:
        image_name = os.path.basename(image_path)
        roi_zip_paths = {
            'cortex_convexhull': os.path.join(area_roi_folder, f'{image_name}cortex_convexhull.zip'),
            'cortex_in': os.path.join(area_roi_folder, f'{image_name}cortex_in.zip'),
            'stele_out': os.path.join(area_roi_folder, f'{image_name}stele_out.zip'),
        }

        process_area_masks(image_path, roi_zip_paths, area_mask_folders, naming_prefix)

def process_project(main_project_folder):
    project_prefix = os.path.basename(os.path.abspath(main_project_folder))
    experiment_folders = [
        os.path.join(main_project_folder, d)
        for d in os.listdir(main_project_folder)
        if os.path.isdir(os.path.join(main_project_folder, d))
    ]

    for exp_folder in experiment_folders:
        print(f"Processing experiment: {exp_folder}")
        process_experiment(exp_folder, project_prefix)

    print("All experiments processed!")

def main():
    if len(sys.argv) < 2:
        print("Please provide the project root path as an argument.")
        print("Usage: python3 mask_gen.py /path/to/project/root")
        sys.exit(1)

    project_root = sys.argv[1]
    if not os.path.isdir(project_root):
        print(f"Error: {project_root} is not a valid directory.")
        sys.exit(1)

    print(f"Starting project processing: {project_root}")
    process_project(project_root)
    print("All done!")

if __name__ == "__main__":
    main()