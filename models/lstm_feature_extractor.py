import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Tuple

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize LSTM feature extractor for sequential stock data.
        
        Args:
            input_size: Number of input features per timestep
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        super(LSTMFeatureExtractor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Feature reduction layer
        self.feature_reduction = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM feature extractor.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Extracted features tensor of shape (batch_size, hidden_size // 4)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_features = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Reduce feature dimensionality
        extracted_features = self.feature_reduction(attended_features)
        
        return extracted_features

class SequentialFeatureProcessor:
    def __init__(self, sequence_length: int = 20):
        """
        Initialize the sequential feature processor.
        
        Args:
            sequence_length: Number of timesteps to consider for each sequence
        """
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def prepare_sequences(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequential data for LSTM processing.
        
        Args:
            df: DataFrame containing the time series data
            feature_columns: List of column names to use as features
            
        Returns:
            Tuple of (sequences, targets) tensors
        """
        # Convert features to numpy array
        features = df[feature_columns].values
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(features) - self.sequence_length):
            seq = features[i:(i + self.sequence_length)]
            target = features[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)
        
        # Convert to tensors
        sequences = torch.FloatTensor(np.array(sequences)).to(self.device)
        targets = torch.FloatTensor(np.array(targets)).to(self.device)
        
        return sequences, targets
    
    def train(self, df: pd.DataFrame, feature_columns: List[str], 
              hidden_size: int = 64, num_layers: int = 2, 
              epochs: int = 50, batch_size: int = 32,
              learning_rate: float = 0.001) -> None:
        """
        Train the LSTM feature extractor.
        
        Args:
            df: DataFrame containing the training data
            feature_columns: List of column names to use as features
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
        """
        # Prepare data
        sequences, targets = self.prepare_sequences(df, feature_columns)
        
        # Initialize model
        input_size = len(feature_columns)
        self.model = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i + batch_size]
                batch_targets = targets[i:i + batch_size]
                
                # Forward pass
                extracted_features = self.model(batch_sequences)
                loss = criterion(extracted_features, batch_targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(sequences):.4f}')
    
    def extract_features(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """
        Extract temporal features from the input data.
        
        Args:
            df: DataFrame containing the data
            feature_columns: List of column names to use as features
            
        Returns:
            Array of extracted features
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Prepare sequences
        sequences, _ = self.prepare_sequences(df, feature_columns)
        
        # Extract features
        self.model.eval()
        with torch.no_grad():
            extracted_features = self.model(sequences)
        
        return extracted_features.cpu().numpy()
    
    def save_model(self, path: str) -> None:
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("No model to save!")
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str, input_size: int, hidden_size: int = 64, num_layers: int = 2) -> None:
        """Load a trained model from a file."""
        self.model = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)
        self.model.load_state_dict(torch.load(path)) 