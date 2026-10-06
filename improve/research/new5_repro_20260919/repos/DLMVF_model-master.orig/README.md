# DLMVF

DLMVF: A deep learning framework based on multi-view fusion for inferring miRNA-drug association. The model integrates various data sources, including drug SMILES sequences, drug molecular graphs, miRNA sequences, gene expression data, and miRNA-drug interactions.


## Key Dependencies

Below are the main libraries and their versions required to run this project. Please fill in the specific versions you are using.

*   **Python**: `3.8`
*   **Torch**: `2.4.0`
*   **torch-geometric**: `2.5.3`
*   **Pandas**: `2.0.3`
*   **RDKit**: `2023.09.06`

## Usage
 **Run the training script:**
    ```bash
    python training.py
    ```

## Project Structure

- `dataprocess.py`: Script for data loading, preprocessing, and feature engineering.
- `model.py`: Contains the PyTorch implementation of the GCNMultiModel.
- `training.py`: Main script to train and evaluate the model.
- `utils.py`: Utility functions used across the project.
- `data/`: Directory for storing all raw and processed data files.
- `Predataprocess/`: Scripts and data related to initial feature processing steps.
