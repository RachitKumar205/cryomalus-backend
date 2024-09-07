# Thermal Anomaly Detection with CNN

This project implements a Convolutional Neural Network (CNN) in PyTorch to detect thermal anomalies in a series of thermal images. The model predicts the coordinates of the anomaly point in the thermal grid.

## Project Structure

- `neural_network_files/train_model.py`: Script to train the CNN model.
- `thermal_cnn.py`: Script to load the trained model and run inference on new data.

## Requirements

- Python 3.x
- PyTorch
- pandas
- numpy
- matplotlib

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/RachitKumar205/thermal-anomaly-detection.git
    cd thermal-anomaly-detection
    ```

2. Install the required packages:
    ```sh
    pip install torch pandas numpy matplotlib
    ```

## Training the Model

1. Prepare your dataset and place it in the `thermal_dataset_output` directory.
2. Run the training script:
    ```sh
    python neural_network_files/train_model.py
    ```
3. The trained model will be saved as `thermal_cnn_model.pth`.

## Running Inference

1. Ensure the trained model `thermal_cnn_model.pth` is in the project directory.
2. Run the inference script:
    ```sh
    python thermal_cnn.py
    ```
3. The script will load the model and predict the anomaly coordinates for a random series from the dataset.

## Functions

### `train_model.py`

- **ThermalCNN**: Defines the CNN architecture.
- **ThermalDataset**: Custom dataset class to load thermal images and labels.
- **Training Loop**: Trains the model and saves it.

### `thermal_cnn.py`

- **load_model**: Loads the trained model.
- **infer**: Runs inference on a given series of thermal images.
- **load_actual_coordinates**: Loads the actual coordinates of the anomaly from metadata.
- **run_inference**: Main function to run inference on a random series.

## Usage

### Training

To train the model, run:
```sh
python neural_network_files/train_model.py
```

### Inference

To run the server, run:
```sh
fastapi dev main.py
```

and access at http://localhost:8000/infer

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.