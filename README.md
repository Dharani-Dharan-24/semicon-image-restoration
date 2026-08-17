# AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Overview
This repository contains our team's solution for the SEMICON India Hackathon 2026. We built an AI model to restore degraded semiconductor inspection images. Our model removes speckle and Gaussian noise while performing 4x super-resolution to recover fine microscopic details, ensuring chip defects are clearly visible.

We utilized the **SwinIR** (Swin Transformer) architecture, fine-tuned specifically for grayscale semiconductor layouts.

## 1. Setup Instructions
To set up the environment and run the code, clone this repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/Dharani-Dharan-24/semicon-image-restoration
cd semicon-image-restoration

# Install the required Python libraries
pip install -r requirements.txt
```

## 2. Model Weights & Restored Outputs (Important Note)
**Note: Due to GitHub's file size limits, the model weights and full restored outputs dataset are hosted on Google Drive. Download them here:**
* **Model Weights (`swinir_model.pth`):** [https://drive.google.com/file/d/16NQBhF-NAL_CvEe42rPb_QUDeFZpehVN/view?usp=drive_link]
* **Restored Output Images (`restored_outputs.zip`):** [https://drive.google.com/file/d/1UMmWR56dkD2pB5wYNh6hlgejJT2F1dzf/view?usp=drive_link]

**Setup for Evaluation:** Once downloaded, please place the `swinir_model.pth` file exactly in this folder path: `weights/swinir_model.pth`

## 3. How to Run the Evaluation Script
The most important file in this repository is `evaluate.py`. It is designed to run without any manual edits.

Run the following command to test the AI on a folder of degraded images:

```bash
python evaluate.py --input_dir /path/to/test_images --output_dir /path/to/save_outputs
```
* `--input_dir`: The path to the folder containing the noisy, low-resolution test images (accepts `.npy`, `.png`, etc.).
* `--output_dir`: The path to the folder where the AI will save the cleaned, high-resolution `.png` images.

## 4. Repository Structure
* `evaluate.py`: The standalone inference script for the benchmarking team.
* `train.py`: The Python script documenting our training process, dataset loaders, loss functions, and optimizer setup.
* `requirements.txt`: The exact Python environment dependencies.
* `weights/download_link.txt`: Direct link to download the model weights.
* `restored_outputs/download_link.txt`: Direct link to download the complete test outputs.
* `models/`: Contains the SwinIR architecture code used by the evaluation script.

## 5. How to Train / Fine-Tune the Model
To fine-tune the model on a custom dataset using transfer learning:
1. Create two folders in the root directory: `train_lq` (for noisy/degraded images) and `train_hq` (for clean/ground-truth images).
2. Ensure the image filenames match exactly across both folders.
3. Run the training script:
   python train.py
This will fine-tune the pre-trained model for 200 epochs and save checkpoints after every epoch in the `weights/` folder.
