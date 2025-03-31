# Multimodal-Bioristor

A deep learning project for multimodal analysis of bioristor sensor data and images.

## Project Overview

This project implements multimodal deep learning models for analysis of bioristor data, which includes:
- Sensor measurements (Rds, DIgs, tds, tgs)
- Multiple images per measurement (N000, N090, NTV, V000, V090, VTV)

The goal is to classify plant health status (healthy, uncertain, stress) using both data modalities.

## Models

### ResNet-LSTM Multimodal Architecture

The main model architecture combines ResNet for image processing and LSTM for sensor time series data:

1. **Image Processing Branch**
   - Custom ResNet implementation (ResNet18 or ResNet34)
   - Processes 6 images per sample
   - Aggregates image features via averaging

2. **Sensor Processing Branch**
   - Bidirectional LSTM network
   - Treats 4 sensor values as a sequence
   - Captures temporal dependencies in sensor readings

3. **Fusion Mechanism**
   - Concatenation of features from both branches
   - Multi-layer classification head
   - Outputs plant health status

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/username/multimodal-bioristor.git
cd multimodal-bioristor
```

2. Install dependencies:
```bash
pip install torch torchvision tqdm pandas pillow scikit-learn matplotlib seaborn
```

## Dataset

The dataset should be organized as:

- `data/mapped_data.csv`: CSV file containing sensor data and image paths
- `data/images/`: Directory containing all images

The CSV file should include columns for:
- Day
- Sensor readings (Rds, DIgs, tds, tgs)
- Image paths (N000, N090, NTV, V000, V090, VTV)
- Plant health label

## Usage

### Training

To train the ResNet-LSTM model:

```bash
python train_resnet_lstm.py \
    --data_csv data/mapped_data.csv \
    --image_dir data/images \
    --batch_size 32 \
    --hidden_size 128 \
    --num_layers 2 \
    --bidirectional \
    --epochs 50 \
    --resnet_type resnet18 \
    --experiment_name "resnet_lstm_experiment"
```

Key parameters:
- `--resnet_type`: Type of ResNet to use (resnet18, resnet34)
- `--hidden_size`: Size of LSTM hidden state
- `--num_layers`: Number of LSTM layers
- `--bidirectional`: Use bidirectional LSTM
- `--freeze_resnet`: Freeze ResNet weights (for transfer learning)

### Testing

To evaluate a trained model:

```bash
python test_resnet_lstm.py \
    --checkpoint_dir checkpoints/resnet_lstm_experiment \
    --data_csv data/mapped_data.csv \
    --image_dir data/images \
    --visualize_features \
    --experiment_name "resnet_lstm_test"
```

The testing script generates:
- Classification report
- Confusion matrix
- Per-class accuracy
- Feature visualizations using t-SNE (if `--visualize_features` is used)

## Feature Visualization

The test script can generate t-SNE visualizations showing the distribution of features for different classes:

1. Image features (from ResNet)
2. Sensor features (from LSTM)
3. Combined features (fusion of both)

This helps understand how well the model separates different classes in the feature space.

## Other Models

The project also includes alternative model architectures:
- Pure ResNet for multimodal processing
- Transformer-based multimodal model
- Custom CNN with LSTM

## License

[MIT License](LICENSE) 