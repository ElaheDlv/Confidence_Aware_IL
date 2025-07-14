# Confidence-Aware Imitation Learning (CARIL) for Autonomous Driving

This repository contains the official implementation of **CARIL: Confidence-Aware Regression in Imitation Learning for Autonomous Driving**. The project introduces a **dual-head neural network** combining **regression** and **classification** for end-to-end vision-based imitation  learning. The classification head provides **confidence scores**, enabling a **correction mechanism** to improve driving stability in **uncertain situations**.

---

## 📑 Table of Contents
1. [Key Contributions](#-key-contributions)
2. [Repository Structure](#-repository-structure)
3. [Prerequisites](#-prerequisites)
4. [Installation & Setup](#-installation--setup)
5. [Workflow Steps](#-workflow-steps)
6. [Running the Simulation](#-running-the-simulation)
7. [Active Inference Agent](#-active-inference-agent)
8. [Usage](#-usage)
9. [Troubleshooting](#-troubleshooting)
10. [Evaluation: Trajectory Similarity](#-evaluation-trajectory-similarity)
11. [Citation](#-citation)
12. [Contact](#-contact)
13. [Future Work](#-future-work)

---

## 🔥 Key Contributions
- **Dual-Head Architecture**: Combines **regression** for precise control and **classification** to estimate confidence.
- **Confidence-Aware Correction**: Applies real-time adjustments when confidence is low, improving driving stability.
- **Closed-Loop Evaluation**: Tested in **CARLA** using controlled scenarios with **predefined routes**.
- **Trajectory Similarity Analysis**: Uses **Frechet distance, Dynamic Time Warping (DTW), and Area Between Curves (ABC)** to assess path accuracy.

---

## 📂 Repository Structure

```
Confidence_Aware_IL/
│── Code/                      # Core scripts for training, evaluation, and model definition
│   ├── data_loader.py         # Data loading functions
│   ├── evaluate.py            # Evaluation script for testing performance
│   ├── model.py               # Model architecture definition
│   ├── train.py               # Training pipeline
│
│── Data Collection/           # Scripts for data collection and preprocessing
│   ├── collect_data.py        # CARLA-based data collection script
│   ├── preprocess_data.py     # Data normalization, augmentation, and scaling
│
│── Driving Evaluation/        # Scripts for testing and post-processing
│   ├── Metric.py              # Metrics for evaluating similarity
│   ├── postprocessing.py      # Post-processing trajectory data
│   ├── test_scenario.py       # CARLA test scenario execution
│
│── results/                   # CSV logs for analysis
│── pretrained_models/         # Directory containing pretrained models
│── reference_data/            # Reference trajectory CSV files
│── .gitignore                 # Ignore unnecessary files
│── README.md                  # This file
```

---

## 🔹 Prerequisites
Before setting up the environment, ensure you have the following installed:
- **Ubuntu 20.04 LTS** (or compatible version)
- **CARLA Simulator** (version 0.9.13+ recommended)
- **Python 3.8+** (recommended via Anaconda environment)
- **Anaconda** (optional but recommended)
- **CUDA Toolkit 11.8** (for GPU support)
- **cuDNN 8.9** (for deep learning acceleration)
- **Git**

---

## 🛠 Installation & Setup

### **1️⃣ Install CARLA Simulator**
CARIL is tested on **CARLA 0.9.15**. Follow the official setup guide:
```bash
# Download CARLA
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.15.tar.gz
mkdir CARLA_0.9.15
cd CARLA_0.9.15

# Extract and Run
tar -xvzf ../CARLA_0.9.15.tar.gz
./CarlaUE4.sh
```

### **2️⃣ Install Dependencies**
Run the following to install required Python packages:
```bash
pip install -r requirements.txt
```

---

## 🔄 Workflow Steps

### **1️⃣ Data Collection**
Collect driving data in CARLA using the data collection script:
```bash
python Data\ Collection/collect_data.py --town Town04
```

### **2️⃣ Data Preprocessing**
Normalize, augment, and filter the collected data:
```bash
python Data\ Collection/preprocess_data.py --input_csv <path_to_input_csv> \
    --output_csv <path_to_output_csv> --input_dir <path_to_images> \
    --output_dir <output_directory>
```
For instance it can be considered as
```bash
python Data\ Collection/preprocess_data.py --input_csv ./output_data/control_commands.csv --output_csv ./output_modified_images/control_commands_modified.csv --input_dir ./output_data/front_rgb/ --output_dir ./output_modified_images/front_rgb/
```
#### **Preprocessing Arguments:**
- `--input_csv`: Path to input CSV file containing raw data.
- `--output_csv`: Path to save the processed CSV file.
- `--input_dir`: Directory containing the input images.
- `--output_dir`: Directory to store processed images and plots.

### **3️⃣ Model Training**
Train the imitation learning model:
```bash
python Code/train.py --data_csv ./output_modified_images/control_commands_modified.csv --epochs 50 --batch_size 32 --lr 0.0001 --image_dir ./output_modified_images/front_rgb/ --experiment_name first_experiment

```

### **4️⃣ Model Evaluation**
Evaluate the trained model on the validation set:
```bash
python Code/evaluate.py --model_path pretrained_models/<experiment_name> --labels_path <path_to_csv> --images_directory <path_to_image>
```
For instance:
```bash
python Code/evaluate.py --model_path pretrained_models/first_experiment_final.h5 --labels_path output_modified_images_test/control_commands_modified.csv --images_directory output_modified_images_test/front_rgb/
```

### **5️⃣ Testing with CARLA**
Test the model in CARLA using predefined driving scenarios:
```bash
python Driving\ Evaluation/test_scenario.py --correction True
```
Logs are saved in `results/`.

### **6️⃣ Post-Processing & Analysis**
Analyze trajectory similarity and performance metrics:
```bash
python Driving\ Evaluation/postprocessing.py
```

---

## 📜 Pretrained Models & Reference Data
- **Pretrained Model:** [Download Here](https://www.dropbox.com/scl/fi/njtnclwcga4sa01sm6aan/Pretrained_model.h5?rlkey=3csab4vz9l4dkh7tvir8hug9a&st=e6x4q9tb&dl=0)
- **Reference Trajectory Data:** [Download Here](https://www.dropbox.com/scl/fi/gjshtuld1p6va3try8q2z/reference_track.csv?rlkey=sgfcbe4p00qh5pqklg8dr5sul&st=fs5dn1gk&dl=0)

---

## 📜 Citation
If you use this repository, please cite:
<!-- 
```bibtex
@inproceedings{Delavari2025CARIL,
  author    = {Elahe Delavari and Aws Khalil and Jaerock Kwon},
  title     = {CARIL: Confidence-Aware Regression in Imitation Learning for Autonomous Driving},
  booktitle = {IROS},
  year      = {2025}
}
```
-->
```bibtex
@misc{delavari2025carilconfidenceawareregressionimitation,
      title={CARIL: Confidence-Aware Regression in Imitation Learning for Autonomous Driving}, 
      author={Elahe Delavari and Aws Khalil and Jaerock Kwon},
      year={2025},
      eprint={2503.00783},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.00783}, 
}
```

---

## 📬 Contact
For questions, please contact [Elahe Delavari](mailto:elahed@umich.edu).

---

### 🚀 Future Work
- **Adaptive Confidence Thresholds** for varying road conditions.
- **Multi-modal Sensor Fusion** integrating LiDAR & radar.
- **Real-world deployment** beyond CARLA simulations.

---

