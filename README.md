# AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Overview
This repository contains our team's solution for the SEMICON India Hackathon 2026. We built an AI model to restore degraded semiconductor inspection images. Our model removes speckle and Gaussian noise while performing 4x super-resolution to recover fine microscopic details, ensuring chip defects are clearly visible.

We utilized the **SwinIR** (Swin Transformer) architecture, fine-tuned specifically for grayscale semiconductor layouts.

## 1. Setup Instructions
To set up the environment and run the code, clone this repository and install the required dependencies:

```bash
# Clone the repository
git clone <https://github.com/Dharani-Dharan-24/semicon-image-restoration>
cd semicon-image-restoration

# Install the required Python libraries
pip install -r requirements.txt
