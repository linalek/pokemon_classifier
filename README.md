# Pokémon Type Classifier: Multi-Stage Deep Learning Challenge
### Universidade Nova de Lisboa - Deep Learning Course (2024-2025)

This repository contains the solution for a Pokémon Image Classification Challenge, exploring various deep learning approaches from a **Multilayer Perceptron (MLP)** to **Convolutional Neural Networks (CNN)** and **Transfer Learning/Fine-Tuning** techniques. It was also a kaggle competition, we get the **Best Private Leaderboard Score Achieved: 0.97 (Task 2)**.



## 1. Project Overview

The core objective was to classify Pokémon images into 9 primary types. The project was structured into three incremental tasks:

* **Task 1: MLP** (**Multilayer Perceptron** (4 layers: 12288 → 256 → 128 → 9 neurons)). Achieved with **0.93 validation accuracy** due to highly effective preprocessing (background removal and image centering).
* **Task 2: CNN from Scratch:** **Custom CNN** (3 Conv layers, 2 FC layers). Achieved with **0.97 private score** and outperformed the MLP, demonstrating superior feature extraction from image data.
* **Task 3: Transfer Learning:** **Pre-trained EfficientNet** with Transfer Learning and Fine-Tuning.Tested against a complex dataset without the custom preprocessing. Fine-Tuning achieved the best performance among the transfer learning models (best F1: **0.2596**).

### 1.1. Critical Preprocessing Step

A major factor in the high performance for the first two tasks was the custom preprocessing pipeline: **undersampling** to balance the dataset, followed by **background removal** (using boundary fixing, thresholding, and morphological operations) and **centering the Pokémon** within the $64\times64$ image. This drastically improved the MLP's validation accuracy from **~0.30 to ~0.93**.


### 1.2. Project Structure

The repository organization reflects the progression of the tasks:

-  `data/`: Contains raw and processed image data for all tasks. 
- `data/Test_processed`, `data/Train_processed`: Processed (centered & background-removed) images used in Tasks 1 & 2.
- `data/Train_task3`, `data/Test_task3`: Image data for the more complex Transfer Learning challenge (Task 3).
- `notebooks/`: Main workspace for model training and analysis. (MLP implementation and initial data preprocessing, Custom CNN implementation and optimization, Transfer Learning and Fine-Tuning pipeline comparison (EfficientNet, ResNet, VGG).
- `train_labels.csv`, `train_labels_task3.csv`: Labels for the respective tasks.
- `slides_presentation/`: Source presentation files.


## 2. Installation and Setup

This project requires Python and standard deep learning libraries, primarily PyTorch.

1.  **Clone the repository:**
    ```bash
    git clone [URL_OF_YOUR_REPOSITORY]
    cd pokemon_classifier
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # .\venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    # Assuming you are using PyTorch and other common packages
    pip install torch torchvision numpy matplotlib scikit-learn pandas jupyter
    ```

---

## 3. How to Run the Code

All training, evaluation, and analysis scripts are contained within the Jupyter Notebooks located in the `notebooks/` directory.

1.  **Start Jupyter Notebook Server:**
    ```bash
    jupyter notebook
    ```
2.  **Navigate to the `notebooks/` directory:**
    * To explore the **MLP** model, open and run `Task1.ipynb`.
    * To explore the **Custom CNN** model, open and run `Task2.ipynb`.
    * To explore the **Transfer Learning** and **Fine-Tuning** experiments, open and run `Task3.ipynb`.

### GPU Usage

The notebooks are configured to automatically check for and use an available **GPU (CUDA)** to accelerate training. This significantly reduces training time.

## 4. Authors

This project was a collaborative effort by **Team LLMΗ** for the Deep Learning Challenge.

* GALLO Lorenzo
* LEKBOURI Lina
* LICHTNER Marc
* WERCK Hugo