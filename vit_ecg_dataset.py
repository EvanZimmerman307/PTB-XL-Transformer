import numpy as np
import scipy.signal
import cv2
import warnings
import torch
from torch.utils.data import Dataset, DataLoader

# --- Configuration Constants ---
SAMPLING_RATE = 100         # Hz
RECORDING_DURATION = 10     # seconds
NUM_LEADS = 12              # Standard 12-lead ECG
TARGET_SIZE = (224, 224)    # H x W for ViT input
HIGH_PASS_CUTOFF = 0.5      # Hz
LOW_PASS_CUTOFF = 40.0      # Hz
NPERSEG = 256               # STFT Window Size
NOVERLAP = int(NPERSEG * 0.75)
NFFT = NPERSEG

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Handle edge cases where a bandpass is not valid, falling back to simpler filters or no filter
    if low >= 1.0 or high >= 1.0 or low >= high:
        if highcut > 0 and lowcut <= 0:
            # Low-pass filter only
            b, a = scipy.signal.butter(order, high, btype='low')
        elif lowcut > 0 and highcut >= nyq:
            # High-pass filter only
            b, a = scipy.signal.butter(order, low, btype='high')
        else:
             warnings.warn(f"Invalid filter parameters: low={low}, high={high}. Returning un-filtered data.")
             return data
    else:
        # Standard bandpass filter
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        
    y = scipy.signal.lfilter(b, a, data)
    return y


def ecg_to_spectrogram(ecg_data: np.ndarray) -> torch.Tensor:
    """
    Transforms a single 12-lead ECG time series (12, 1000) into a 12-channel
    spectrogram image (12, 224, 224) as a PyTorch Tensor.
    """
    spectrograms = []
    
    # Ensure input is float for correct processing
    ecg_data = ecg_data.astype(np.float32)

    for lead_signal in ecg_data:
        # 1. Signal Cleaning and Filtering
        filtered_signal = butter_bandpass_filter(
            lead_signal,
            lowcut=HIGH_PASS_CUTOFF,
            highcut=LOW_PASS_CUTOFF,
            fs=SAMPLING_RATE
        )

        # 2. Time-Frequency Transformation (STFT)
        f, t, Sxx = scipy.signal.stft(
            filtered_signal,
            fs=SAMPLING_RATE,
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            nfft=NFFT,
            window='hann',
            return_onesided=True
        )
        
        # Log-scale magnitude (Decibels)
        magnitude_spectrogram = np.abs(Sxx)
        log_spectrogram = 10 * np.log10(magnitude_spectrogram + 1e-6)

        # 3. Normalization and Resizing
        min_val = np.min(log_spectrogram)
        max_val = np.max(log_spectrogram)
        
        if max_val > min_val:
            normalized_spec = (log_spectrogram - min_val) / (max_val - min_val)
        else:
            normalized_spec = np.zeros_like(log_spectrogram)

        resized_spec = cv2.resize(
            normalized_spec,
            TARGET_SIZE,
            interpolation=cv2.INTER_LINEAR
        )

        spectrograms.append(resized_spec)

    # 4. Construct 12-Channel Tensor (C, H, W)
    final_input_tensor = np.stack(spectrograms, axis=0)
    
    # Convert final NumPy array to PyTorch Tensor
    return torch.from_numpy(final_input_tensor).float()

class ECGDataset(Dataset):
    """
    A PyTorch Dataset that loads raw ECG data and performs the
    spectrogram preprocessing on-the-fly in the __getitem__ method.
    """
    def __init__(self, raw_ecg_data, labels):
        """
        Initializes the dataset.

        Args:
            raw_ecg_data (np.ndarray): The full dataset of raw ECGs, shape (N, 12, 1000).
            labels (np.ndarray): The corresponding labels, shape (N, num_classes).
        """
        # Store references to the raw data and labels.
        # This is where the memory efficiency comes from: we store the small raw data.
        self.raw_ecg_data = raw_ecg_data
        self.labels = labels

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.raw_ecg_data)

    def __getitem__(self, idx):
        """
        Retrieves one sample, applies the full preprocessing pipeline, and returns
        the spectrogram and its label.
        """
        # 1. Get the raw ECG signal and label for the current index
        raw_signal = self.raw_ecg_data[idx]
        label = self.labels[idx]

        # 2. Perform the intensive preprocessing transformation
        spectrogram_tensor = ecg_to_spectrogram(raw_signal)

        # Convert label to tensor (assuming float for multi-label classification)
        label_tensor = torch.from_numpy(label).float()
        
        # Returns (C, H, W) tensor and the label tensor
        return spectrogram_tensor, label_tensor