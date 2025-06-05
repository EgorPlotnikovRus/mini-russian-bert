import argparse
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
import torch.nn.functional as F
import mlflow.pytorch

from get_masked_dataset import get_dataloaders
from load_config import get_config
from get_models import get_model


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config yaml file')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
args = parser.parse_args()

params = get_config(args.config)


gradient_accumulation_steps = params["train"]["gradient_accumulation_steps"]

mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
mlflow.set_experiment(params["mlflow"]["experiment_name"])

dataset = load_from_disk(params["data"]["dataset_path"])
tokenizer = AutoTokenizer.from_pretrained(params["tokenizer"])

train_dataloader, val_dataloader = get_dataloaders(dataset, tokenizer,
                                                   train_size=params["data"]["train_size"],
                                                   val_size=params["data"]["val_size"],
                                                   train_batch_size=params["train"]["train_batch_size"],
                                                   val_batch_size=params["eval"]["val_batch_size"],
                                                   mlm_probability=params["data"]["mlm_probability"])

model = get_model(params=params['model'],
                  tokenizer=tokenizer,
                    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


mlflow.start_run(run_name=params["mlflow"]["run_name"])
mlflow.log_params(params)


optimizer = AdamW(model.parameters(), lr=params["train"]["lr"], betas=(params["train"]["b_1"], params["train"]["b_2"]), weight_decay=params["train"]["weight_decay"])

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=(params["train"]["num_warmup_steps"] // (params["train"]["train_batch_size"] * params["train"]["gradient_accumulation_steps"])),
    num_training_steps=len(train_dataloader) * params["train"]["epoches"] // gradient_accumulation_steps
)

scaler = torch.cuda.amp.GradScaler()

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['epoch'], checkpoint['global_step']

def save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, global_step):
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, checkpoint_path)

start_epoch = 0
global_step = 0

if args.checkpoint:
    start_epoch, global_step = load_checkpoint(args.checkpoint, model, optimizer, scheduler, scaler)
    print(f"Resuming training from epoch {start_epoch}, global step {global_step}")

window_size = 50
loss_history = []
mlm_loss_history = []

def validate(model, val_dataloader, device, tokenizer):
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0

    with torch.no_grad():
        for val_batch in tqdm(val_dataloader, desc="Validation"):
            input_ids = val_batch["input_ids"].to(device)
            labels = val_batch["labels"].to(device)

            outputs = model(input_ids)
            logits = outputs.logits

            mlm_mask = (input_ids == tokenizer.mask_token_id)
            if mlm_mask.any():
                mlm_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1))[mlm_mask.view(-1)],
                    labels.view(-1)[mlm_mask.view(-1)]
                )
            else:
                mlm_loss = torch.tensor(0.0, device=device)

            total_val_loss += mlm_loss.item()
            num_val_batches += 1

    return {
        "val_loss": total_val_loss / num_val_batches
    }

mlm_loss_avg = 1.0
beta_s = 0.99

for epoch in range(params["train"]["epoches"]):
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=True)

    for i, batch in enumerate(progress_bar):
        if epoch == start_epoch and i < global_step:
            continue

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)


        with torch.cuda.amp.autocast():
            #outputs = model(input_ids)
            #logits = outputs.logits
            logits = model(input_ids).logits

            mlm_mask = (input_ids == tokenizer.mask_token_id)
            mlm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1))[mlm_mask.view(-1)],
                labels.view(-1)[mlm_mask.view(-1)]
            ) if mlm_mask.any() else torch.tensor(0.0, device=device)


            loss = mlm_loss / gradient_accumulation_steps


        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params["train"].get("max_grad_norm", 1.0))

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        if (i + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % (params["train"]["log_interval"] // (params["train"]["train_batch_size"] * params["train"]["gradient_accumulation_steps"])) == 0:
                mlflow.log_metric("train_loss", total_loss / num_batches, step=global_step)
                mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=global_step)

            if global_step % (params['train']['val_interval'] // (params["train"]["train_batch_size"] * params["train"]["gradient_accumulation_steps"])) == 0:
                val_metrics = validate(model, val_dataloader, device, tokenizer)
                mlflow.log_metric("val_loss", val_metrics["val_loss"], step=global_step)
                print(f"\nStep {global_step} | Val Loss: {val_metrics['val_loss']:.4f}")

            if global_step % (params["train"]["save_interval"] // (params["train"]["train_batch_size"] * params["train"]["gradient_accumulation_steps"])) == 0:
                checkpoint_path = f"./checkpoints/checkpoint_step_{global_step}.pt"
                save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, global_step)
                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

        progress_bar.set_postfix({
            "Train": f"{total_loss / num_batches:.3f}",
            "LR": f"{scheduler.get_last_lr()[0]:.6f}"
        })

    if num_batches > 0:
        checkpoint_path = f"./checkpoints/checkpoint_epoch_{epoch}.pt"
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, global_step)
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

        mlflow.log_metric("train_loss_epoch", total_loss / num_batches, step=global_step)

        print(
            f"Epoch {epoch + 1} | Train Loss: {total_loss / num_batches:.4f}")

mlflow.pytorch.log_model(model, "model")
mlflow.end_run()