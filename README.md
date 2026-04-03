# Overview
IEEE TGCN paper "Learning-based Channel Estimation and Beamforming Framework for Battery-Free Backscatter Communications" code

'''UNDER CONSTRUCTION'''

## Prerequisites
Before running the code, ensure you have all the necessary dependencies installed. You can install them using `pip` with the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## How to Run

### Training
You can start the training process by executing the `train.sh` script. This script requires a GPU device ID as an argument to specify which GPU to use.

```bash
# Usage: ./train.sh (GPU_ID)
./train.sh 0
```
*(Replace `0` with the specific GPU index you wish to utilize.)*

## Citation
If you find this code useful for your research, please consider citing our paper:

```bibtex
@ARTICLE{BackCom_RACE,
  author={Shin, Jaemin and Kim, Yusung},
  journal={IEEE Transactions on Green Communications and Networking}, 
  title={Learning-Based Channel Estimation and Beamforming Framework for Battery-Free Backscatter Communications}, 
  year={2026},
  volume={10},
  number={},
  pages={2418-2431},
  keywords={Channel estimation;Array signal processing;Antennas;Backscatter;Transmitting antennas;Radio frequency;Estimation;Discrete Fourier transforms;Internet of Things;Vectors;Internet of Things;backscatter communication;neural network;transformer encoder;beamforming;channel estimation},
  doi={10.1109/TGCN.2026.3670371}}
```
```
