import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
from torchvision.transforms import PILToTensor


class CellDataset(Dataset):
    def __init__(self, root_path, split='train', mask_types=['AR_mask']):
        assert split in ['train', 'val']
        for mt in mask_types:
            assert mt in ['AR_mask', 'Endoderm_mask', 'Cortex_mask']

        self.root_path = root_path
        self.mask_types = mask_types
        self.is_multi_class = len(mask_types) > 1
        self.images, self.masks = [], []
        self.label_map = {mt: i + 1 for i, mt in enumerate(mask_types)}

        # Mapping for internal suffix stripping
        suffix_map = {
            'AR_mask': '_image_mask_ar',
            'Endoderm_mask': '_image_mask_endoderm',
            'Cortex_mask': '_image_mask_cortex',
        }

        for subfolder in sorted(os.listdir(root_path)):
            subpath = os.path.join(root_path, subfolder)
            if not os.path.isdir(subpath): continue

            samples = sorted([
                d for d in os.listdir(subpath)
                if os.path.isdir(os.path.join(subpath, d))
            ])

            total = len(samples)
            train_end = int(0.85 * total)
            chosen = samples[:train_end] if split == 'train' else samples[train_end:]

            for sample in chosen:
                sample_path = os.path.join(subpath, sample)
                img_dir = os.path.join(sample_path, 'Original_images')
                if not os.path.exists(img_dir): continue

                image_list = sorted([
                    os.path.join(img_dir, f) for f in os.listdir(img_dir)
                    if f.lower().endswith('.tif')
                ])

                if self.is_multi_class:
                    self._process_multiclass_masks(sample_path, image_list, suffix_map)
                else:
                    self._process_binary_masks(sample_path, image_list, suffix_map)

        print(f"\nFinal dataset ({split} split, masks={mask_types}): "
              f"{len(self.images)} images, {len(self.masks)} masks, "
              f"mode={'multi-class' if self.is_multi_class else 'binary'}")

        # transforms
        target_size = (341, 512)
        # target_size = (543, 817)
        #target_size = (681, 1024)
        # target_size = (1088, 1636)
        


        self.img_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        if self.is_multi_class:
            self.mask_transform = transforms.Compose([
                transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
                PILToTensor(),
                transforms.Lambda(lambda x: x.squeeze(0).long())
            ])
        else:
            self.mask_transform = transforms.Compose([
                transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x > 0).long().squeeze(0))
            ])

    def _normalize_basename(self, filename):
        base = os.path.basename(filename)
        return base.replace('.tif.tif', '.tif').replace('.tif', '')

    def _get_base_filename(self, mask_filepath, suffix_to_remove):
        base = self._normalize_basename(mask_filepath)
        if suffix_to_remove and suffix_to_remove in base:
            base = base.replace(suffix_to_remove, '')
        return base + '.tif'

    def _process_binary_masks(self, sample_path, image_list, suffix_map):
        mask_type = self.mask_types[0]
        mask_dir = os.path.join(sample_path, mask_type)
        if not os.path.isdir(mask_dir): return

        suffix = suffix_map.get(mask_type, '')
        mask_files = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
            if f.lower().endswith('.tif')
        ])
        mask_map = {
            self._get_base_filename(m, suffix): m for m in mask_files
        }

        for img in image_list:
            name = os.path.basename(img)
            if name in mask_map:
                self.images.append(img)
                self.masks.append(mask_map[name])
            else:
                print(f"No binary match for {name}")

    def _process_multiclass_masks(self, sample_path, image_list, suffix_map):
        maps = {}
        for mt in self.mask_types:
            md = os.path.join(sample_path, mt)
            if os.path.isdir(md):
                suffix = suffix_map[mt]
                mf = sorted([
                    os.path.join(md, f) for f in os.listdir(md)
                    if f.lower().endswith('.tif')
                ])
                mapping = {
                    self._get_base_filename(m, suffix): m for m in mf
                }
                maps[mt] = mapping

        for img in image_list:
            name = os.path.basename(img)
            combo = []
            ok = True
            for mt in self.mask_types:
                if name in maps.get(mt, {}):
                    combo.append((maps[mt][name], self.label_map[mt]))
                else:
                    print(f"Missing {mt} mask for {name}")
                    ok = False
                    break
            if ok:
                self.images.append(img)
                self.masks.append(combo)

    def _load_combined_mask(self, mask_info):
        combined = None
        for mp, lbl in mask_info:
            m = Image.open(mp).convert('L')
            arr = np.array(m)
            arr = (arr > 0).astype(np.uint8) 
            if combined is None:
                combined = np.zeros_like(arr, dtype=np.uint8)
            combined[arr == 1] = lbl 
        return Image.fromarray(combined)


    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        if self.is_multi_class:
            mask = self._load_combined_mask(self.masks[idx])
        else:
            mask = Image.open(self.masks[idx]).convert('L')

        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        return {"pixel_values": img, "labels": mask.long()}

    def __len__(self):
        return len(self.images)
