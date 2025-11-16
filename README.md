
## Installation 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md). 

## Data Preparation
Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.


## Evaluation 
Please refer to the [EVAL.md](docs/EVAL.md) for detailed instructions on using the evaluation scripts and reproducing the official results using our pre-trained models.

## Training 
Please refer to the [TRAIN.md](docs/TRAIN.md) for detailed instructions on training PromptSRC and IVLP baseline from scratch.


# Làm theo các bước sau:
## Create Dockerfile
## build Image containers
```bash
sudo docker build -t promptmeo_rtx5090 .
```
## Run containers
```bash
sudo docker run --gpus all -it --rm   --ipc=host   --name promptmeo_runner   -v $(pwd):/workspace
/PromptMeo   promptmeo_rtx5090
```
sudo docker run --gpus all -it --rm \
  --ipc=host \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e TORCH_CUDA_ARCH_LIST="9.0" \
  -v $(pwd):/workspace/PromptMeo \
  promptmeo_rtx5090