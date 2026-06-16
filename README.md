# EnvSocial-Diff

## EnvSocial-Diff: A Diffusion-Based Crowd Simulation Model with Environmental Conditioning and Individual-Group Interaction
Bingxue Zhao, Qi Zhang*, Hui Huang, EnvSocial-Diff: A Diffusion-Based Crowd Simulation Model with Environmental Conditioning and Individual-Group Interaction, ICLR 2026


## Abstract
Modeling realistic pedestrian trajectories requires accounting for both social interactions and environmental context, yet most existing approaches largely emphasize social dynamics. We propose EnvSocial-Diff: a diffusion-based crowd  simulation model informed by social physics and augmented with environmental conditioning and individual–group interaction. Our structured environmental conditioning module explicitly encodes obstacles, objects of interest, and lighting levels, providing interpretable signals that capture scene constraints and attractors. In parallel, the individual–group interaction module goes beyond individual-level modeling by capturing both fine-grained interpersonal relations and group-level conformity through a graph-based design. Experiments on multiple benchmark datasets demonstrate that EnvSocial-Diff outperforms the latest state-of-the-art methods, underscoring the importance of explicit environmental conditioning and multi-level social interaction for realistic crowd simulation.

## Overview
We release the PyTorch code for our ICLR 2026 method, a comprehensive trajectory prediction model with promising performance on UCY, GC datasets.

## Dependencies
### Core framework
```bash
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric==2.3.1
```
### Key research libraries
```bash
pip install salesforce-lavis==1.0.2  # For multimodal components
pip install einops==0.6.1 timm==0.4.12 transformers==4.26.1
```
### Utilities
```bash
pip install omegaconf opencv-python tqdm matplotlib
```

Alternatively, you can install all dependencies via:
```bash
 pip install -r requirements.txt
  ```

## Training
### Example: Training on UCY dataset
```bash
python run.py --config configs/ucy.yaml
```
### GPU Selection
By default, the model is hardcoded to run on `cuda:0`. If you need to use a different GPU, please ensure consistency between the code and the configuration file:

1. **In `run.py`**: Modify **line 15** to your target device ID:
   ```python
   torch.cuda.set_device(1)  # Change 0 to your target GPU ID
2. **In `configs/ucy.yaml`**: Ensure the settings in your YAML file are updated accordingly to match the hardware changes.

## Test
To evaluate a trained model, use the `test.py` script. You need to provide the path to the configuration file and the pre-trained model checkpoint.
```bash
# General command
python test.py --config [PATH_TO_CONFIG] --data_dict_path [PATH_TO_DATA] --model_name [PATH_TO_CHECKPOINT]
```

## Checkpoints
To run the training or testing scripts, please ensure the following model files are placed in the `checkpoints/` directory:


checkpoints/

├── bert-base-uncased/ [👉 Click Here to Download](https://huggingface.co/bert-base-uncased)   
└── resnet50-0676ba61.pth [👉 Click Here to Download](https://download.pytorch.org/models/resnet50-0676ba61.pth)

## Data & Pre-trained Models
Due to GitHub's file size limits, datasets and pre-trained weights are hosted on Google Drive.
* **Download Link**: [👉 Click Here to Download](https://drive.google.com/file/d/1CXnWmP9up_lzgRUpXd-1bebYp8b9wdsz/view?usp=sharing)
## Contributing
If you’d like to contribute to EnvSocial-Diff, please fork the repository and submit a pull request. We welcome contributions that enhance the functionality and usability of the project.

## Acknowledgement
This work was supported in parts by National Key R&D Program of China (2024YFB3908500, 2024YFB3908504), NSFC (62202312), Guangdong Basic and Applied Basic Research Foundation(2023B1515120026), Shenzhen Science and Technology Program (KQTD20210811090044003), Teaching Reform Key Program (JG2024018) and Scientific Foundation for Youth Scholars and Scientific Development Funds from Shenzhen University.


## Reference
```bash
@inproceedings{
zhao2026envsocialdiff,
title={EnvSocial-Diff: A Diffusion-Based Crowd Simulation Model with Environmental Conditioning and Individual- Group Interaction},
author={Bingxue Zhao and Qi Zhang and Hui Huang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=2XBAm3Dbnt}
}
```

## Contact
For any inquiries, please contact [zqyq](mailto:zqyq@example.com). 
