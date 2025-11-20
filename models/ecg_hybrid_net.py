import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import tqdm
import numpy as np
import math

# Experiment 4: class balanced loss + d_model=256 (was 128) + num_layers=4 (was 2), dropout = 0.3 (was 0.1)
# smaller learning rate (e.g., 1e-4 → 5e-5)
class ECGHybridNet(nn.Module):
    def __init__(self, input_dim=12, num_classes=5, d_model=128, nhead=8, num_layers=2, dropout=0.4):
        super().__init__()
        # --- CNN backbone ---
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),  # Added layer
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),  # 2x downsample
            nn.Conv1d(128, d_model, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )
        
        # Experiment 3 - not helpful, try again for experiment 7 (model now had extra conv layer & max pool)
        # self.positional_encoding = PositionalEncoding(d_model=128)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Classification head (attention pooling) ---
        self.attention_pool = nn.Linear(d_model, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )


    def forward(self, x):                 # x: (B, C, L)
        x = self.cnn(x)                   # -> (B, D, L') L' is modified by stride
        x = x.transpose(1, 2)             # -> (B, seq_len, d_model)
        # x = self.positional_encoding(x)   # -> (B, seq_len, d_model), Experiment 3 & 7
        x = self.transformer(x)           # -> (B, L', D)

        # Attention pooling
        attn_weights = F.softmax(self.attention_pool(x), dim=1)  # (B, L, 1)
        x = (x * attn_weights).sum(dim=1)  # (B, D)

        return self.classifier(x)

    def fit(self, X_train, y_train, X_val, y_val, model_path, num_epochs=50, batch_size=32, patience=10):
        if torch.cuda.is_available():
            print("Using GPU")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # Convert to tensors and transpose X to (N, C, L)
        X_train = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2)  # Assuming X_train is (N, L, C), make (N, C, L)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32).transpose(1, 2)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        # TRAINING RUN 2: Class Balanced Loss
        class_freq = y_train.sum(dim=0)
        pos_weight = (y_train.shape[0] - class_freq) / class_freq

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss, optimizer, scheduler
        # criterion = nn.BCEWithLogitsLoss() # TRAINING RUN 1
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.clone().detach().to(device)) # TRAINING RUN 2
        optimizer = AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4) # Experiment 4: was lr=1e-4, weight_decay=1e-3
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        scaler = torch.amp.GradScaler()

        best_val_loss = float('inf')
        patience_counter = 0
        best_model = None

        for epoch in tqdm.tqdm(range(num_epochs), desc="Epochs"):
            self.train()
            running_train_loss = 0.0

            for X, y in tqdm.tqdm(train_loader, desc="Train Batches", leave=False):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda'):
                    outputs = self(X)
                    loss = criterion(outputs, y)

                scaler.scale(loss).backward()
                clip_grad_norm_(self.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                running_train_loss += loss.item() * X.size(0)

            avg_train_loss = running_train_loss / len(train_loader.dataset)

            # Validation
            self.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self(X)
                        loss = criterion(outputs, y)
                    running_val_loss += loss.item() * X.size(0)

            avg_val_loss = running_val_loss / len(val_loader.dataset)

            scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model = self.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Save best model
        self.load_state_dict(best_model)
        torch.save(best_model, model_path + 'best.pth')
        print(f"Best model saved to {model_path}best.pth")

    def predict(self, X_split, batch_size=32):
        # X_split: (N, L, C) numpy array
        # Assumes model is on device and in eval mode, but to be safe:
        self.eval()
        device = next(self.parameters()).device

        X_tensor = torch.tensor(X_split, dtype=torch.float32).transpose(1, 2)  # (N, C, L)
        dataset = TensorDataset(X_tensor, torch.zeros(len(X_tensor)))  # Dummy targets for DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for X_batch, _ in dataloader:
                X_batch = X_batch.to(device)
                logits = self(X_batch)  # (B, num_classes)
                probs = torch.sigmoid(logits)  # multilabel probabilities
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)  # (N, num_classes)

# Experiment 3 adding positional encodings
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=0.1)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # Add batch dimension for broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, d_model)
        Returns:
            Tensor, shape (batch_size, seq_len, d_model) with positional encodings added
        """
        # Add positional encoding to the input embeddings
        # The .pe tensor is sliced to match the sequence length of x
        x = x + self.pe[:, :x.size(1)] 
        return self.dropout(x)