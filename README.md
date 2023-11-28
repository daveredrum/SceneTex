# SceneTex: High-Quality Texture Synthesis for Indoor Scenes via Diffusion Priors

<p align="center"><img src="docs/static/teaser/teaser.jpg" width="100%"/></p>

## Introduction

We propose SceneTex, a novel method for effectively generating high-quality and style-consistent textures for indoor scenes using depth-to-image diffusion priors. 
Unlike previous methods that either iteratively warp 2D views onto a mesh surface or distillate diffusion latent features without accurate geometric and style cues, SceneTex formulates the texture synthesis task as an optimization problem in the RGB space where style and geometry consistency are properly reflected. 
At its core, SceneTex proposes a multiresolution texture field to implicitly encode the mesh appearance. 
We optimize the target texture via a score-distillation-based objective function in respective RGB renderings. 
To further secure the style consistency across views, we introduce a cross-attention decoder to predict the RGB values by cross-attending to the pre-sampled reference locations in each instance.
SceneTex enables various and accurate texture synthesis for 3D-FRONT scenes, demonstrating significant improvements in visual quality and prompt fidelity over the prior texture generation methods.

## Setup

The code is tested on Ubuntu 20.04 LTS with PyTorch 1.12.1 CUDA 11.3 installed. Please follow the following steps to install PyTorch first. To run our method, you should at least have a NVIDIA GPU with 48 GB RAM (NVIDIA RTX A6000 works for us).

```shell
# create and activate the conda environment
conda create -n scenetex python=3.9
conda activate scenetex``

# install PyTorch 2.0.1
```shell
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Then, install PyTorch3D:

```shell
# install runtime dependencies for PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub

# install PyTorch3D
conda install pytorch3d -c pytorch3d
```

Install `xformers` to accelerate transformers:

```shell
# please don't use pip to install it, as it only supports PyTorch>=2.0
conda install xformers -c xformers
```

Install tinycudann:

```shell
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Install the necessary packages listed out in requirements.txt:

```shell
pip install -r requirements.txt
```

### 3D-FRONT scenes

Please download the [3D-FRONT scenes](https://www.dropbox.com/scl/fi/ql0dfgglw14puwr2opyeo/3D-FRONT_preprocessed.zip?rlkey=4d2hxgl8cyi0g0dqrbc6axsn4&dl=0), and unzip it under `data/`. You should be able to see the preprocessed data constructed as follows:

```
data/
    |-- 3D-FRONT_preprocessed/
        |-- scenes/
            |-- <scene_id>
                |-- <room_id>
                    |-- meshes/
                    |-- cameras/
```

## Usage

To make sure everything is set up and configured correctly, you can run the following script to generate texture for a 3D-FRONT scene.

```shell
./bash/bohemian.sh
```

All generated assets should be found under `outputs`. To configure the style or the target scene, you can run the following script:

```shell
stamp=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir="outputs/" # your output dir
prompt="a bohemian style living room" # your prompt
scene_id="93f59740-4b65-4e8b-8a0f-6420b339469d/room_4" # preprocessed scene

python python scripts/train_texture.py --config config/template.yaml --stamp $stamp --log_dir $log_dir --prompt "$prompt" --scene_id "$scene_id"
```

We provide a template file for all critical parameters in `config/template.yaml`. Please take a look at it in case you want to further costumize the optimization process.

## Citation

If you found our work helpful, please kindly cite this papers:

```bibtex

```

## Acknowledgement

We would like to thank [yuanzhi-zhu/prolific_dreamer2d](https://github.com/yuanzhi-zhu/prolific_dreamer2d) for providing such a great and powerful codebase for variational score distillation.

## License

SceneTex is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License](LICENSE).

Copyright (c) 2023 Dave Zhenyu Chen, Haoxuan Li, Hsin-Ying Lee, Sergey Tulyakov, and Matthias Nießner
