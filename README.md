# Hybrid CNN Architectures for Keystroke Prediction from Surface Electromyography

CS247A Final Project – Winter 2026  
Emma Thorssell, Caleb Kim, Nikhita Kalluri, Madison Sarmiento

## Overview

This project investigates deep learning architectures for decoding typing keystrokes from **surface electromyography (sEMG)** signals.

Using the **emg2qwerty dataset**, we train models to map continuous EMG signals recorded from wrist electrodes to character sequences using **Connectionist Temporal Classification (CTC)**.

The main goal is to minimize **Character Error Rate (CER)** when predicting typed characters from EMG signals recorded from a **single participant**.

We evaluate several hybrid architectures:
- CNN + Vanilla RNN
- CNN + GRU
- CNN + BiLSTM
- CNN + TCN + BiLSTM
- CNN + Transformer + BiLSTM (exploratory)

Our best performing model achieved:

**Test CER = 18.67**  
using an optimized **CNN-BiLSTM architecture**.

---

## Repository Structure

```text
emg2qwerty/
├── emg2qwerty/
│   ├── lightning.py        # PyTorch Lightning modules
│   ├── modules.py          # Model architectures
│   ├── data.py             # Dataset loading utilities
│   └── transforms.py       # Data preprocessing
│
├── config/
│   ├── model/
│   ├── user/
│   └── trainer/
│
├── scripts/
│   ├── generate_splits.py
│   └── print_dataset_stats.py
│
├── models/
│   └── checkpoints/
│
├── environment.yml
└── README.md
```

---
# emg2qwerty Dataset
[ [`Paper`](https://arxiv.org/abs/2410.20081) ] [ [`Dataset`](https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz) ] [ [`Blog`](https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024/) ] [ [`BibTeX`](#citing-emg2qwerty) ]

We use the **emg2qwerty dataset** introduced by:

> Sivakumar et al., 2024  
> *emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography*

The dataset contains:

- **108 users**
- **346 hours of recordings**
- **32 EMG channels** (16 per wrist)
- **2 kHz sampling rate**
- aligned **QWERTY keystrokes**

For this project we train models on the **single-user subset**:
User ID: 89335547


Each recording is stored as an **HDF5 session file** containing:
- left/right EMG signals
- typed prompts
- ground truth keystrokes
- timestamps

---

## Setup

```shell
# Install [git-lfs](https://git-lfs.github.com/) (for pretrained checkpoints)
git lfs install

# Clone the repo, setup environment, and install local package
git clone git@github.com:joe-lin-tech/emg2qwerty.git ~/emg2qwerty 
cd ~/emg2qwerty
conda env create -f environment.yml
conda activate emg2qwerty
pip install -e .

# Download the dataset, extract, and symlink to ~/emg2qwerty/data
cd ~ && wget https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
tar -xvzf emg2qwerty-data-2021-08.tar.gz
ln -s ~/emg2qwerty-data-2021-08 ~/emg2qwerty/data
```
---

# Training Models
### Personalized (single-user) model

```shell
python -m emg2qwerty.train
user="single_user"
trainer.accelerator=gpu
trainer.devices=1
```

All models were trained with:
- up to **80 epochs**
- **early stopping** based on validation CER
- **CTC loss**
- fixed random seed for reproducibility

---

# Architectures Evaluated
### Baseline
**TDS Convolutional Encoder**
Temporal Depthwise Separable convolution layers followed by a linear classifier trained with **CTC loss**.

Baseline performance:
Test CER = 21.89


---

### Hybrid CNN Models

Hybrid architectures combine:

- CNN feature extraction
- temporal sequence models

| Model | Test CER |
|------|------|
| Baseline CNN | 21.89 |
| CNN + Vanilla RNN | 20.92 |
| CNN + GRU | 21.44 |
| CNN + BiLSTM | 20.66 |
| CNN + TCN + BiLSTM | 21.37 |

The **CNN-BiLSTM** model was selected for further optimization.

---

# Optimization Experiments

Several techniques were evaluated to improve performance.

| Method | Test CER |
|------|------|
| CNN-BiLSTM (chosen baseline) | 20.66 |
| Dropout after CNN layers | 19.08 |
| Dropout after CNN + LSTM layers | 20.42 |
| Scaling MLP + LSTM features | **18.00** |
| Filters | 20.96 |
| Normalization | 18.67 |

Best configuration:

CNN-BiLSTM with scaled hidden features and dropout after CNN layers.


Final model performance:
Test CER = 18.67

---

# TCN Experiments

We also explored replacing the TDS block with a **Temporal Convolutional Network (TCN)**.

| Kernel | Dilation | Test CER |
|------|------|------|
| 3 | exponential | 21.37 |
| 5 | exponential | 21.72 |
| 3 | linear | failed |
| 3 | cyclic | 99.74 |

TCNs did not outperform the baseline architecture.

---

# Key Findings
1. **Hybrid convolutional-recurrent architectures outperformed convolution models.**
2. **CNN + BiLSTM performed best for EMG decoding.**
3. **Transformers perform poorly due to long input sequences and limited training data.**
4. **Channel interactions are important for EMG signals**, which may explain why TDS and CNN layers perform well.

---

# Key Files Modified
- emg2qwerty/modules.py
- emg2qwerty/lightning.py

### Model Architecture
- emg2qwerty/modules.py
- emg2qwerty/lightning.py


### Data preprocessing
- emg2qwerty/transforms.py


### Hyperparameters
- config/model/*.yaml


### Dataset splits
- config/user/single_user.yaml

---

# License
emg2qwerty is CC-BY-NC-4.0 licensed, as found in the LICENSE file.

# Citing emg2qwerty
```shell
@misc{sivakumar2024emg2qwertylargedatasetbaselines,
      title={emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography},
      author={Viswanath Sivakumar and Jeffrey Seely and Alan Du and Sean R Bittner and Adam Berenzweig and Anuoluwapo Bolarinwa and Alexandre Gramfort and Michael I Mandel},
      year={2024},
      eprint={2410.20081},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.20081},
}
```


