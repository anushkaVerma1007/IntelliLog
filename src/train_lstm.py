# src/train_lstm.py
import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from template_extractor import normalize_msg

# --- 1. Define the DeepLog-inspired LSTM Model Architecture ---
class DeepLogLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, num_layers=2):
        super(DeepLogLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # Input shape: [batch_size, seq_len]
        embeds = self.embedding(x)           # Shape: [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embeds)      # Shape: [batch_size, seq_len, hidden_dim]
        # Take the output of the final time step
        last_step_out = lstm_out[:, -1, :]  # Shape: [batch_size, hidden_dim]
        logits = self.fc(last_step_out)      # Shape: [batch_size, vocab_size]
        return logits

# --- 2. Custom PyTorch Dataset for Log Sequences ---
class LogSequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# --- 3. Sequence Preprocessing and Vocabulary Builder ---
def prepare_sequences(df, window_size=3):
    """
    Groups logs chronologically, maps them to vocabulary IDs, and builds
    sliding window sequences and targets for sequential model training.
    """
    # Build vocabulary: map unique normalized templates to integer IDs
    # Reserve ID 0 for padding, ID 1 for unknown templates
    unique_templates = sorted(df['norm_msg'].unique())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, template in enumerate(unique_templates, start=2):
        vocab[template] = idx

    # Map the dataset to IDs
    event_ids = [vocab.get(msg, 1) for msg in df['norm_msg']]

    # Build sliding window sequences
    sequences = []
    targets = []
    
    if len(event_ids) <= window_size:
        # Fallback if log sequence is too short: pad the sequence
        pad_len = window_size + 1 - len(event_ids)
        event_ids = [0] * pad_len + event_ids

    for i in range(len(event_ids) - window_size):
        sequences.append(event_ids[i : i + window_size])
        targets.append(event_ids[i + window_size])

    return np.array(sequences), np.array(targets), vocab

# --- 4. Main Training Routine ---
def train_lstm(parsed_pkl_path='data/processed/parsed.pkl', epochs=50, batch_size=8, window_size=3):
    print("--- Starting LSTM (DeepLog) Model Training ---")

    if not os.path.exists(parsed_pkl_path):
        print(f"Error: Processed data file not found at '{parsed_pkl_path}'.")
        print("Please run 'python src/dataset_builder.py' first.")
        return

    # Load parsed log DataFrame
    df = pd.read_pickle(parsed_pkl_path)
    df['norm_msg'] = df['norm_msg'].fillna('')

    print(f"Loaded {len(df)} parsed logs for sequence training.")

    # Prepare input-output sequences
    sequences, targets, vocab = prepare_sequences(df, window_size=window_size)
    vocab_size = len(vocab)
    print(f"Log event vocabulary size: {vocab_size}")
    print(f"Generated {len(sequences)} sequences of length {window_size} for training.")

    # Create DataLoader
    dataset = LogSequenceDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")

    # Initialize deep network
    model = DeepLogLSTM(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64, num_layers=2)
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for seqs, targs in dataloader:
            seqs, targs = seqs.to(device), targs.to(device)

            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, targs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * seqs.size(0)
        
        avg_loss = epoch_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Training Loss: {avg_loss:.4f}")

    # Save trained model and vocabulary mapping
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save vocabulary
    vocab_path = os.path.join(models_dir, 'lstm_vocab.joblib')
    joblib.dump(vocab, vocab_path)
    print(f"Saved vocabulary to: {vocab_path}")

    # Save model weights
    model_path = os.path.join(models_dir, 'lstm_anomaly_detector.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved LSTM model to: {model_path}")
    print("--- LSTM Model Training Completed ---")


# --- 5. Anomaly Inference Utility ---
def detect_sequence_anomalies(log_messages, window_size=3, top_k=2):
    """
    Given a list of log messages, normalizes them, maps them to IDs,
    and uses the trained LSTM model to flag out-of-sequence events.
    """
    models_dir = 'models'
    vocab_path = os.path.join(models_dir, 'lstm_vocab.joblib')
    model_path = os.path.join(models_dir, 'lstm_anomaly_detector.pth')

    if not (os.path.exists(vocab_path) and os.path.exists(model_path)):
        return [False] * len(log_messages)

    vocab = joblib.load(vocab_path)
    vocab_size = len(vocab)

    # Reconstruct the model structure and load state
    model = DeepLogLSTM(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Preprocess inputs
    normalized_msgs = [normalize_msg(msg) for msg in log_messages]
    event_ids = [vocab.get(msg, 1) for msg in normalized_msgs] # Default to <UNK> if template is unseen

    anomalies = []
    # For the first few elements up to the window size, we cannot build a sequence.
    # We default them to False (no anomaly).
    for i in range(min(window_size, len(event_ids))):
        anomalies.append(False)

    # Predict sequence anomalies
    with torch.no_grad():
        for i in range(window_size, len(event_ids)):
            seq = event_ids[i - window_size : i]
            seq_tensor = torch.tensor([seq], dtype=torch.long)
            
            # Predict logits for the next event
            logits = model(seq_tensor)
            probabilities = torch.softmax(logits, dim=1).numpy()[0]
            
            # DeepLog rule: If the actual next event index is NOT in the top K predicted events,
            # it is considered an out-of-order sequence violation (anomaly).
            actual_next_idx = event_ids[i]
            top_k_indices = np.argsort(probabilities)[-top_k:]
            
            is_seq_anomaly = actual_next_idx not in top_k_indices
            anomalies.append(is_seq_anomaly)

    return anomalies

if __name__ == '__main__':
    # Train the DeepLog model
    train_lstm(epochs=50)
    
    # Test anomaly detection inference helper
    test_logs = [
        "User 'alice' logged in successfully from 192.168.1.10",
        "Transaction initiated for user 'alice' amount 55.00",
        "Query executed: SELECT * FROM users WHERE id=12",
        "Exception: Unable to connect to the database master node." # Strange sequence
    ]
    results = detect_sequence_anomalies(test_logs)
    print("\n--- Sequence Inference Test ---")
    for log, is_anom in zip(test_logs, results):
        print(f"Anomaly: {is_anom} | Log: {log}")
