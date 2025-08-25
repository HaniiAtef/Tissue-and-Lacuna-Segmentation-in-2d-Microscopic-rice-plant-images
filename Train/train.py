import os
import argparse
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import (
    AutoConfig,
    TrainingArguments,
    Trainer,
    SegformerForSemanticSegmentation,
    EvalPrediction
)

from sklearn.metrics import precision_score, recall_score, jaccard_score
from dataset import CellDataset
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter


# === ARGUMENT PARSER ===
parser = argparse.ArgumentParser(description="Train SegFormer with combined loss")

parser.add_argument("--model", type=str, choices=['b3', 'b4'], help='Choose a model (b3 or b4)', required=True)
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--ce_weight", type=float, default=1.0, help="Weight for CrossEntropy loss")
parser.add_argument("--dice_weight", type=float, default=1.0, help="Weight for Dice loss")
parser.add_argument("--mask_type", type=str, nargs="+", default=["Endoderm_mask", "Cortex_mask"],
                    help="List of mask types (e.g., --mask_type AR_mask or --mask_type Cortex_mask Endoderm_mask)")

args = parser.parse_args()

# === TIMESTAMP ===
job_id = os.environ.get("SLURM_JOB_ID", "nojob")
timestamp = datetime.now().strftime('%d-%m-%Y')

# === PATHS ===
root_path = "/lustre/fswork/projects/rech/duc/utv28ua/task1/Curated_dataset_train"

if args.model == 'b3':
    model_name = "/lustre/fswork/projects/rech/duc/utv28ua/task1/code/models/segformer_b3_model"
    if args.mask_type == ['AR_mask']:
        scratch_root = "/lustre/fswork/projects/rech/duc/utv28ua/task1/output_AR/segformer_output_b3"
        log_path = "/lustre/fswork/projects/rech/duc/utv28ua/task1/output_AR/tensorboard_b3"
        run_name = f"BCE_{args.ce_weight}_Dice_{args.dice_weight}_lr_{args.lr}_B3_{job_id}_{timestamp}"
    else:
        scratch_root = "/lustre/fswork/projects/rech/duc/utv28ua/task1/output_Cortex_Endoderm/segformer_output_b3"
        log_path = "/lustre/fswork/projects/rech/duc/utv28ua/task1/output_Cortex_Endoderm/tensorboard_b3"
        run_name = f"CE_{args.ce_weight}_Dice_{args.dice_weight}_lr_{args.lr}_B3_{job_id}_{timestamp}"

elif args.model == 'b4':
    model_name = "/lustre/fswork/projects/rech/duc/utv28ua/task1/code/models/segformer-offline2"
    if args.mask_type == ['AR_mask']:
        scratch_root = "/lustre/fswork/projects/rech/duc/utv28ua/task1/output_AR/segformer_output_b4"
        log_path = "/lustre/fswork/projects/rech/duc/utv28ua/task1/output_AR/tensorboard_b4"
        run_name = f"BCE_{args.ce_weight}_Dice_{args.dice_weight}_lr_{args.lr}_B4_{job_id}_{timestamp}"
    else:
        scratch_root = "/lustre/fswork/projects/rech/duc/utv28ua/task1/output_Cortex_Endoderm/segformer_output_b4"
        log_path = "/lustre/fswork/projects/rech/duc/utv28ua/task1/output_Cortex_Endoderm/tensorboard_b4"
        run_name = f"CE_{args.ce_weight}_Dice_{args.dice_weight}_lr_{args.lr}_B4_{job_id}_{timestamp}"

output_dir = os.path.join(scratch_root, f"segformer_output_{run_name}")
log_dir = os.path.join(log_path, f"{run_name}")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# === DATASETS ===
train_dataset = CellDataset(root_path, split='train', mask_types=args.mask_type)
val_dataset = CellDataset(root_path, split='val', mask_types=args.mask_type)

# === COLLATE FUNCTION ===
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch])
    }

class OptimizedDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs = torch.softmax(inputs, dim=1)
        
        batch_size, num_classes, h, w = inputs.shape
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
   
        intersection = torch.sum(inputs * targets_one_hot, dim=(2, 3))
        union = torch.sum(inputs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# === CUSTOM SEGFORMER MODEL ===
class CustomSegformer(SegformerForSemanticSegmentation):
    def __init__(self, config, ce_weight=1.0, dice_weight=1.0):
        super().__init__(config)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = config.num_labels
        
        if self.num_classes == 2:  
            self.classification_loss = nn.BCEWithLogitsLoss()
            self.is_binary = True
            print("Using BCE loss for binary classification")
        else: 
            self.classification_loss = nn.CrossEntropyLoss()
            self.is_binary = False
            print(f"Using CrossEntropy loss for {self.num_classes}-class classification")
            
        self.dice_loss = OptimizedDiceLoss()

    def forward(self, pixel_values=None, labels=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        outputs = super().forward(pixel_values=pixel_values, labels=labels, **kwargs)
        
        if labels is not None:
            logits = outputs.logits
            if logits.shape[-2:] != labels.shape[-2:]:
                labels = F.interpolate(labels.unsqueeze(1).float(), size=logits.shape[-2:], mode='nearest').squeeze(1).long()

            if self.is_binary:
                binary_labels = (labels > 0).float()  
               
                positive_logits = logits[:, 1:2, :, :]
                loss_ce = self.classification_loss(positive_logits, binary_labels.unsqueeze(1))
            else:
                loss_ce = self.classification_loss(logits, labels)
            
            loss_dice = self.dice_loss(logits, labels)
            outputs.loss = self.ce_weight * loss_ce + self.dice_weight * loss_dice

        return outputs

# === FASTER METRICS COMPUTATION ===
def compute_metrics_fast(eval_pred: EvalPrediction):
    """Optimized metrics computation"""
    logits, labels = eval_pred
    
  
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):  
        labels = labels.cpu().numpy()
    
   
    preds = np.argmax(logits, axis=1)
    
    
    if preds.shape[-2:] != labels.shape[-2:]:
        
        preds = torch.from_numpy(preds)
        preds = F.interpolate(preds.unsqueeze(1).float(), size=labels.shape[-2:], mode='nearest').squeeze(1).long()
        preds = preds.numpy()
    

    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    
    
    if len(preds_flat) > 100000:
        sample_size = min(100000, len(preds_flat))
        idx = np.random.choice(len(preds_flat), sample_size, replace=False)
        preds_flat = preds_flat[idx]
        labels_flat = labels_flat[idx]

    return {
        "precision": precision_score(labels_flat, preds_flat, average="macro", zero_division=1),
        "recall": recall_score(labels_flat, preds_flat, average="macro", zero_division=1), 
        "iou": jaccard_score(labels_flat, preds_flat, average="macro", zero_division=1),
    }


class FastEpochTensorBoardCallback(TrainerCallback):
    def __init__(self, trainer, train_sample_size=1000):
        self.trainer = trainer
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.train_sample_size = train_sample_size  # Sample only 1000 examples for train metrics
        
        # Store training metrics during the epoch
        self.epoch_train_losses = []
        self.current_epoch_start_step = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Reset metrics collection for new epoch"""
        self.epoch_train_losses = []
        self.current_epoch_start_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Collect training loss during epoch"""
        if logs and "train_loss" in logs:
            self.epoch_train_losses.append(logs["train_loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        
    
        print(f"Computing training metrics for epoch {epoch} (sampling {self.train_sample_size} examples)...")
        
        # Create a small subset of training data for metrics
        train_indices = torch.randperm(len(self.trainer.train_dataset))[:self.train_sample_size]
        train_subset = torch.utils.data.Subset(self.trainer.train_dataset, train_indices)
        
        # Evaluate on this small subset
        train_metrics = self.trainer.evaluate(
            eval_dataset=train_subset,
            metric_key_prefix="train"
        )
        
        # === VALIDATION METRICS (FULL DATASET) ===
        print(f"Computing validation metrics for epoch {epoch}...")
        val_metrics = self.trainer.evaluate(
            eval_dataset=self.trainer.eval_dataset,
            metric_key_prefix="eval"
        )
        
      
        
        # Training metrics (4 graphs)
        if "train_loss" in train_metrics:
            self.tb_writer.add_scalar("epoch_metrics/train_loss", train_metrics["train_loss"], epoch)
        if "train_precision" in train_metrics:
            self.tb_writer.add_scalar("epoch_metrics/train_precision", train_metrics["train_precision"], epoch)
        if "train_recall" in train_metrics:
            self.tb_writer.add_scalar("epoch_metrics/train_recall", train_metrics["train_recall"], epoch)
        if "train_iou" in train_metrics:
            self.tb_writer.add_scalar("epoch_metrics/train_iou", train_metrics["train_iou"], epoch)
            
        # Validation metrics (4 graphs)  
        if "eval_loss" in val_metrics:
            self.tb_writer.add_scalar("epoch_metrics/eval_loss", val_metrics["eval_loss"], epoch)
        if "eval_precision" in val_metrics:
            self.tb_writer.add_scalar("epoch_metrics/eval_precision", val_metrics["eval_precision"], epoch)
        if "eval_recall" in val_metrics:
            self.tb_writer.add_scalar("epoch_metrics/eval_recall", val_metrics["eval_recall"], epoch)
        if "eval_iou" in val_metrics:
            self.tb_writer.add_scalar("epoch_metrics/eval_iou", val_metrics["eval_iou"], epoch)
        
        
        self.tb_writer.flush()
        
        print(f"Epoch {epoch} metrics logged to TensorBoard")

    def on_train_end(self, args, state, control, **kwargs):
        self.tb_writer.close()

# === CONFIG + MODEL ===
num_classes = len(args.mask_type) + 1  
config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
model = CustomSegformer.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True,
    ce_weight=args.ce_weight,
    dice_weight=args.dice_weight,
)

# === OPTIMIZED TRAINING ARGUMENTS ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=12,      
    per_device_eval_batch_size=8,        
    gradient_accumulation_steps=2,     
    learning_rate=args.lr,
    num_train_epochs=50,
    warmup_steps=50,
    lr_scheduler_type="constant",
    eval_strategy="epoch",              
    save_strategy="epoch",               
    logging_dir=log_dir,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,         
    metric_for_best_model="eval_iou",   
    greater_is_better=True,
    fp16=True,
    fp16_full_eval=True,
    weight_decay=0.01,
    report_to="tensorboard",
    dataloader_num_workers=8,            
    dataloader_pin_memory=True,
    eval_accumulation_steps=2,           
    remove_unused_columns=False,
    include_inputs_for_metrics=False,    
)

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics_fast,  # Use the optimized version
)

# === ADD THE FAST TENSORBOARD CALLBACK ===
trainer.add_callback(FastEpochTensorBoardCallback(trainer, train_sample_size=1000))

# === TRAIN ===
print("Starting training...")
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Number of classes: {num_classes}")
print(f"Mask types: {args.mask_type}")
print(f"Output directory: {output_dir}")
print(f"TensorBoard logs: {log_dir}")

trainer.train()
trainer.save_model(output_dir)

print("Training completed!")
print(f"Model saved to: {output_dir}")