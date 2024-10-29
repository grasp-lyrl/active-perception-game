# An Active Perception Game for Robust Information Gathering
[Paper](https://arxiv.org/abs/2404.00769), [Video](https://www.youtube.com/watch?v=zzvhCXsdn7o)

## Abstract
Active perception approaches select future viewpoints by using some estimate of the information gain. An inaccurate estimate can be detrimental in critical situations, e.g., locating a person in distress. However the true information gained can only be calculated post hoc, i.e., after the observation is realized. We present an approach for estimating the discrepancy between the information gain (which is the average over putative future observations) and the true information gain. The key idea is to analyze the mathematical relationship between active perception and the estimation error of the information gain in a game-theoretic setting. Using this, we develop an online estimation approach that achieves sub-linear regret (in the number of time-steps) for the estimation of the true information gain and reduces the sub-optimality of active perception systems.

## Setup
### Installation
```
# clone repo
git clone git@github.com:grasp-lyrl/active-perception-game.git

# set up environment
conda create -n apg python=3.9 cmake=3.14.0 -y
conda activate apg
python -m pip install --upgrade pip

# habitat
conda install habitat-sim=0.2.5 withbullet -c conda-forge -c aihabitat -y

# install PyTorch 2.0.1 with CUDA 11.8:
pip uninstall torch torchvision functorch tinycudann
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# install tiny-cuda-nn
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# requirements for rotorpy
pip install -e planning/rotorpy

# other requirements
conda install scikit-image PyYAML imageio tqdm scipy rich
pip install lpips opencv-python seaborn
```
### Download habitat data
```
# extract to data/scene_datasets/
https://drive.google.com/file/d/1qXl0iTlKawCXpJ1QJDM-IljmlUVXuyNp/view?usp=drive_link

# you can do so by gdown
pip install gdown==4.6.0
mkdir -p data/scene_datasets/
cd data/scene_datasets/
gdown https://drive.google.com/uc?id=1qXl0iTlKawCXpJ1QJDM-IljmlUVXuyNp
unzip hssd-hab.zip
```

### Run pipeline
```
# scene 1
python scripts/pipeline.py --sem-num 29 --habitat-scene 102816036

# scene 2
python scripts/pipeline.py --sem-num 29 --habitat-scene 102344250

# scene 3
python scripts/pipeline.py --sem-num 29 --habitat-scene 102344280

# scene 4
python scripts/pipeline.py --sem-num 29 --habitat-scene 102344529
```
To run the baseline method, i.e. without our improvement algorithm, replace `scripts/pipeline.py` by `scripts/pipeline_orig.py`.

Data will be saved in `data/habitat_collection/` with the following format:
```
/timestamp
	/checkpoints: the saved NeRF model checkpoints
	/maps: the 2D occupancy maps from NeRF
	/test: save test data
	/train: save train data collected during active perception
	/vis: images, predictions, fpv, and tpv for videos
	errors.npy stores the evaluation errors during active perception
	uncertainty.npy stores the predictive information during active perception
```

## Citation
```
@article{he2024active,
  title={An Active Perception Game for Robust Autonomous Exploration},
  author={He, Siming and Tao, Yuezhan and Spasojevic, Igor and Kumar, Vijay and Chaudhari, Pratik},
  journal={arXiv preprint arXiv:2404.00769},
  year={2024}
}
```