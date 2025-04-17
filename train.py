import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import BertTokenizer
from models.vqa_model import VQAModel
from data.dataset import ECGVQADataset
from utils.helpers import load_config
import time
from torch.cuda.amp import autocast, GradScaler

# Enable benchmark mode in cudnn for performance boost if input sizes do not change
torch.backends.cudnn.benchmark = True

config = load_config("configs/config.yaml")
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = ECGVQADataset(config['train_json'], config['image_dir'], tokenizer, config['max_seq_length'])

# Optimized DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=4,          # Adjust based on your CPU cores
    pin_memory=True         # Faster transfer to CUDA
)

model = VQAModel(config).to(device)
optimizer = AdamW(model.parameters(), lr=config['learning_rate'])  # More efficient optimizer
criterion = CrossEntropyLoss()

scaler = GradScaler()  # For mixed precision training
model.train()

start_time = time.time()

for epoch in range(config['num_epochs']):
    epoch_start_time = time.time()
    total_loss = 0
    epoch_samples = len(train_loader)

    for batch_idx, (image, input_ids, attention_mask, label) in enumerate(train_loader):
        image = image.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            output = model(image, input_ids, attention_mask)
            loss = criterion(output, label)

        # Scaled backward pass and step
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Logging per batch
        batch_progress = (batch_idx + 1) / epoch_samples * 100
        elapsed_time = time.time() - start_time
        epoch_elapsed_time = time.time() - epoch_start_time
        time_per_batch = epoch_elapsed_time / (batch_idx + 1)
        remaining_time = (epoch_samples - (batch_idx + 1)) * time_per_batch

        print(f"Epoch {epoch + 1}/{config['num_epochs']} | Batch {batch_idx + 1}/{epoch_samples} | "
              f"Loss: {loss.item():.4f} | Progress: {batch_progress:.2f}% | "
              f"Elapsed Time: {epoch_elapsed_time:.2f}s | Time Remaining: {remaining_time:.2f}s")

    # Epoch summary
    epoch_elapsed_time = time.time() - epoch_start_time
    total_elapsed_time = time.time() - start_time
    avg_epoch_time = total_elapsed_time / (epoch + 1)
    remaining_epochs = config['num_epochs'] - (epoch + 1)
    remaining_time = avg_epoch_time * remaining_epochs

    print(f"Epoch {epoch + 1} completed. Total Loss: {total_loss:.4f}")
    print(f"Epoch {epoch + 1} completed. Elapsed Time: {epoch_elapsed_time:.2f}s | "
          f"Total Elapsed Time: {total_elapsed_time:.2f}s | Estimated Time Remaining: {remaining_time:.2f}s")

torch.save(model.state_dict(), config['model_save_path'])
print("Model saved.")
