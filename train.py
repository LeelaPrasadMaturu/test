import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from transformers import BertTokenizer
from models.vqa_model import VQAModel
from data.dataset import ECGVQADataset
from utils.helpers import load_config
import time
import os
from torch.cuda.amp import autocast, GradScaler

# Enable performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

config = load_config("configs/config.yaml")
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

# Checkpoint configuration
checkpoint_path = config.get('checkpoint_path', 'checkpoint.pth')
last_checkpoint = None if not os.path.exists(checkpoint_path) else checkpoint_path

# Initialize components
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = ECGVQADataset(config['train_json'], config['image_dir'], tokenizer, config['max_seq_length'])

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=min(8, os.cpu_count()),  # Dynamic workers based on CPU count
    pin_memory=True,
    persistent_workers=True  # Reduces worker initialization overhead
)

model = VQAModel(config).to(device)
optimizer = AdamW(model.parameters(), lr=config['learning_rate'], fused=True)  # Use fused optimizer
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])  # Add learning rate scheduler
criterion = CrossEntropyLoss()
scaler = GradScaler()

# Load checkpoint if available
start_epoch = 0
if last_checkpoint:
    checkpoint = torch.load(last_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scaler.load_state_dict(checkpoint['scaler_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch + 1}")

# Compile model for PyTorch 2.x performance boost (optional)
if torch.__version__ >= "2.0":
    model = torch.compile(model)

def save_checkpoint(epoch):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'loss': total_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved after epoch {epoch + 1}")

model.train()
start_time = time.time()
log_interval = 10  # Reduce logging frequency

for epoch in range(start_epoch, config['num_epochs']):
    epoch_start_time = time.time()
    total_loss = 0
    epoch_samples = len(train_loader)

    for batch_idx, (image, input_ids, attention_mask, label) in enumerate(train_loader):
        # Asynchronous data transfer
        image = image.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # Gradient accumulation preparation
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        # Mixed precision forward
        with autocast():
            output = model(image, input_ids, attention_mask)
            loss = criterion(output, label) / config['grad_accum_steps']  # If using gradient accumulation

        # Scaled backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation check
        if (batch_idx + 1) % config['grad_accum_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config['grad_accum_steps']

        # Reduced logging frequency
        if batch_idx % log_interval == 0:
            batch_progress = (batch_idx + 1) / epoch_samples * 100
            epoch_elapsed = time.time() - epoch_start_time
            batch_time = epoch_elapsed / (batch_idx + 1e-8)
            remaining = (epoch_samples - batch_idx) * batch_time
            
            print(f"Epoch {epoch + 1:03d} | Batch {batch_idx + 1:04d}/{epoch_samples} | "
                  f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Remaining: {remaining:.1f}s")

    # Epoch statistics
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / epoch_samples
    lr = scheduler.get_last_lr()[0]
    
    print(f"Epoch {epoch + 1} Summary | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | LR: {lr:.2e}")

    # Save checkpoint after each epoch
    save_checkpoint(epoch)

# Final model save
torch.save(model.state_dict(), config['model_save_path'])
print("Training complete. Model saved.")
