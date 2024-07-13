# DPMesh: Exploiting Diffusion Prior for Occluded Human Mesh Recovery (CVPR 2024)
*[Yixuan Zhu\*](https://eternalevan.github.io/), [Ao Li\*](https://rammusleo.github.io/), [Yansong Tang†](https://andytang15.github.io/), [Wenliang Zhao](https://wl-zhao.github.io/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)*
----
[**[Paper]**](https://arxiv.org/abs/2404.01424) | 
[**[Project_page]**](https://rammusleo.github.io/dpmesh-proj/)

The repository contains the official implementation for the paper "DPMesh: Exploiting Diffusion Prior for Occluded Human Mesh Recovery" (CVPR 2024).

DPMesh is an innovative framework for occluded human <ins>**Mesh**</ins> recovery that capitalizes on the profound <ins>**D**</ins>iffusion <ins>**P**</ins>rior about object structure and spatial relationships embedded in a pre-trained text-to-image diffusion model.
## 📋 To-Do List

* [x] Release model and inference code.
* [ ] Release code for training dataloader .

## 💡 Pipeline

![](./assets/pipeline.png)

## ⭐️ Performance

![](./assets/performance.png)
![](./assets/table.png)

## 😀Quick Start
### ⚙️ 1. Installation

We recommend you to use an [Anaconda](https://www.anaconda.com/) virtual environment. If you have installed Anaconda, run the following commands to create and activate a virtual environment.
``` bash
conda env create -f environment.yaml
conda activate dpmesh
pip install git+https://github.com/cloneofsimo/lora.git
```
### 💾 2. Data Preparation

We prepare the data in a samilar way as [3DCrowdNet](https://github.com/hongsukchoi/3DCrowdNet_RELEASE) & [JOTR](https://github.com/hongsukchoi/3DCrowdNet_RELEASE/blob/main/assets/directory.md). Please refer to [here](https://github.com/hongsukchoi/3DCrowdNet_RELEASE/blob/main/assets/directory.md) for *dataset*, *SMPL model*, *VPoser model*. 

For 3DPW-OC and 3DPW-PC, we apply the same input key-points annotations as [JOTR](https://github.com/hongsukchoi/3DCrowdNet_RELEASE/blob/main/assets/directory.md). Please refer to [3DPW-OC](https://drive.google.com/file/d/1IPE8Yw7ysd97Uv6Uw24el1yRs2r_HtCR/view?usp=sharing) & [3DPW-PC](https://drive.google.com/file/d/1xzZvUj1lR1ECbzUI4JOooC_r2LF6Qs5m/view?usp=sharing).

**For evaluation only, you can just prepare 3DPW dataset (images and annotations) and the joint regressors**, we provide the directory structure below.

```
|-- common
|   |-- utils
|   |   |-- human_model_files
|   |   |-- smplpytorch
|-- data 
|   |-- J_regressor_extra.npy 
|   |-- 3DPW
|   |   |-- 3DPW_latest_test.json
|   |   |-- 3DPW_oc.json
|   |   |-- 3DPW_pc.json
|   |   |-- 3DPW_validation_crowd_hhrnet_result.json
|   |   |-- imageFiles
|   |   |-- sequenceFiles
|   |-- Human36M  
|   |   |-- J_regressor_h36m_correct.npy
|   |-- MSCOCO  
|   |   |-- J_regressor_coco_hip_smpl.npy
```


### 🗂️ 3. Download Checkpoints

Please download our pretrained checkpoints from [this link](https://cloud.tsinghua.edu.cn/d/1d6cd3ee30204bb59fce/) and put them under `./checkpoints`. The file directory should be:

```
|-- checkpoints
|--|-- 3dpw_best_ckpt.pth.tar
|--|-- 3dpw-crowd_best_ckpt.pth.tar
|--|-- 3dpw-oc_best_ckpt.pth.tar
|--|-- 3dpw-pc_best_ckpt.pth.tar
```

### 📊 4. Evaluation

You can evaluate DPMesh use following commands:

```bash
CUDA_VISIBLE_DEVICES=0 \
torchrun \
--master_port 29591 \
--nproc_per_node 1 \
eval.py \
--cfg ./configs/main_train.yml \
--exp_id="main_train" \
--distributed \
```

The evaluation process can be done with one Nvidia GeForce RTX 4090 GPU (24GB VRAM). You can use more GPUs by specifying the GPU ids.

### 🎨 5. Rendering

You can render the video with the human mesh on it. First of all, you may follow these steps below to pre-process your video data.

1. Create a directory in `./demo` and save your video in `./demo/testvideo/testvideo.mp4`
    ```
    |-- demo
    |--|-- testvideo
    |--|--|-- annotations
    |--|--|-- images
    |--|--|-- renderimgs
    |--|--|-- testvideo.mp4
    ```
2. Split your video into image frames (using [ffmpeg](https://ffmpeg.org/)) and save the images in `./demo/testvideo/images`
3. Using the off-the-shelf detectors like [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) to fetch the keypoints of each person in each image, saving the `.pkl` results in `./demo/testvideo/annotations`. You can use whatever detectors you like, but please pay attention to the key-points format. Here we use [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) format, which has 17 key-points of each person.
4. Setting the `renderimg = 'testvideo'` in `config.py` and choose the pre-trained model to resume (like 3dpw_best_ckpt.pth.tar).

Now you can run `render.py` to render the human mesh:
```bash
CUDA_VISIBLE_DEVICES=0 \
torchrun \
--master_port 29591 \
--nproc_per_node 1 \
render.py \
--cfg ./configs/main_train.yml \
--exp_id="render" \
--distributed \
```

Finally you can combine the rendered images in `./demo/testvideo/renderimgs` together into a video (you can also use [ffmpeg](https://ffmpeg.org/)). 


## 🫰 Acknowledgments

We would like to express our sincere thanks to the author of [JOTR](https://github.com/xljh0520/JOTR) for the clear code base and quick response to our issues. 

We also thank [ControlNet](https://github.com/lllyasviel/ControlNet), [VPD](https://github.com/wl-zhao/VPD) and [LoRA](https://github.com/cloneofsimo/lora), for our code is partially borrowing from them.

## Q & A
1. If you find an error as below, please refer to [this link](https://stackoverflow.com/questions/65637222/runtimeerror-subtraction-the-operator-with-a-bool-tensor-is-not-supported) for help.
```bash
RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.
```


## 🔖 Citation
```
@article{zhu2024dpmesh,
  title={DPMesh: Exploiting Diffusion Prior for Occluded Human Mesh Recovery},
  author={Zhu, Yixuan and Li, Ao and Tang, Yansong and Zhao, Wenliang and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:2404.01424},
  year={2024}
}
```

## 🔑 License

This code is distributed under an [MIT LICENSE](./LICENSE).
