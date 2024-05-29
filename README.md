# Deform3DGS: Flexible Deformation for Fast Surgical Scene Reconstruction with Gaussian Splatting

Official code implementation for [Deform3DGS](https://arxiv.org/abs/2405.17835), a Gaussian Splatting based framework for surgical scene reconstruction.

<!--### [Project Page]() -->
> [Deform3DGS: Flexible Deformation for Fast Surgical Scene Reconstruction with Gaussian Splatting](https://arxiv.org/abs/2405.17835) \
> Shuojue Yang, Qian Li, Daiyun Shen, Bingchen Gong, Qi Dou, Yueming Jin \
> MICCAI2024, **Early Accept**

## Demo: reconstruction within 1 minute

https://github.com/jinlab-imvr/Deform3DGS/assets/157268160/d58deb50-36ce-4cde-9e65-e3ce8bb851dc

Compared to previous SOTA method in fast reconstruction, our method reduces the training time to **1 minute** for each clip in EndoNeRF dataset, demonstrating remarkable superiority in efficiency.


## Abstract
<!--![](assets/overview.png)-->
<p align="center">
  <img src="assets/overview.png" />
</p>

Tissue deformation poses a key challenge for accurate surgical scene reconstruction. Despite yielding high reconstruction quality, existing methods suffer from slow rendering speeds and long training times, limiting their intraoperative applicability. Motivated by recent progress in 3D Gaussian Splatting, an emerging technology in real-time 3D rendering, this work presents a novel fast reconstruction framework, termed **Deform3DGS**, for deformable tissues during endoscopic surgery. Specifically, we introduce 3D GS into surgical scenes by integrating a point cloud initialization to improve reconstruction. Furthermore, we propose a novel flexible deformation modeling scheme (FDM) to learn tissue deformation dynamics at the level of individual Gaussians. Our FDM can model the surface deformation with efficient representations, allowing for real-time rendering performance. More importantly, FDM significantly accelerates surgical scene reconstruction, demonstrating considerable clinical values, particularly in intraoperative settings where time efficiency is crucial. Experiments on DaVinci robotic surgery videos indicate the efficacy of our approach, showcasing superior reconstruction fidelity PSNR: (37.90) and rendering speed (338.8 FPS) while substantially reducing training time to only 1 minute/scene.

## Environment setup
Tested with an Ubuntu workstation  , 4090GPU.

```bash
git clone https://github.com/jinlab-imvr/Deform3DGS.git
cd Deform3DGS
conda create -n Deform3DGS python=3.7 
conda activate Deform3DGS

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

## Datasets
We use 6 clips from [EndoNeRF](https://github.com/med-air/EndoNeRF) and 3 clips manually extracted from [StereoMIS](https://zenodo.org/records/7727692) to verify our method. 

To use the two available examples in [EndoNeRF](https://github.com/med-air/EndoNeRF) dataset. Please download the data via [this link](https://forms.gle/1VAqDJTEgZduD6157) and organize the data according to the [guideline](https://github.com/med-air/EndoNeRF.git).

To use the [StereoMIS](https://zenodo.org/records/7727692) dataset, please follow this [github repo](https://github.com/aimi-lab/robust-pose-estimatot.git) to preprocess the dataset and organize the depth, masks, images, intrinsic and extrinsic parameters in the same format as [EndoNeRF](https://github.com/med-air/EndoNeRF). In our implementation, we used [RAFT](https://github.com/princeton-vl/RAFT) to estimate the stereo depth for [StereoMIS](https://zenodo.org/records/7727692) clips.

The data structure is as follows:
```
data
| - endonerf_full_datasets
|   | - cutting_tissues_twice
|   |   | -  depth/
|   |   | -  images/
|   |   | -  masks/
|   |   | -  pose_bounds.npy 
|   | - pushing_soft_tissues
| - StereoMIS
|   | - stereo_seq_1
|   | - stereo_seq_2
```


## Training
To train Deform3DGS with customized hyper-parameters, please make changes in `arguments/endonerf/default.py`. 

To train Deform3DGS, run the following example command:
```
python train.py -s data/endonerf_full_datasets/pulling_soft_tissues --expname endonerf/pulling_fdm --configs arguments/endonerf/default.py 
```

## Testing
We use the same testing pipeline with [EndoGaussian](https://github.com/yifliu3/EndoGaussian/tree/master) to perform rendering and evaluation seperately.

### Rendering
To run the following example command to render the images:

```
python render.py --model_path output/endonerf/pulling_fdm  --skip_train --reconstruct_test --configs arguments/endonerf/default.py
```
Please follow [EndoGaussian](https://github.com/yifliu3/EndoGaussian/tree/master) to skip rendering. Of note, you can also set `--reconstruct_train`, `--reconstruct_test`, and `--reconstruct_video` to reconstruct and save the `.ply` 3D point cloud of the rendered outputs for  `train`, `test` and`video` sets, respectively.

### Evaluation
To evaluate the reconstruction quality, run following command:

```
python metrics.py --model_path output/endonerf/pulling_fdm -p test
```
Note that you can set `-p video`, `-p test`, `-p train` to select the set for evaluation.

## Acknowledgements
This repo borrows some source code from [EndoGaussian](https://github.com/yifliu3/EndoGaussian/tree/master), [4DGS](https://github.com/hustvl/4DGaussians), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), and [EndoNeRF](https://github.com/med-air/EndoNeRF). We would like to acknowledge these great prior literatures for inspiring our work.

Huge thanks to [EndoGaussian](https://github.com/yifliu3/EndoGaussian/tree/master) for their great and timely effort in releasing the framework adapting Gaussian Splatting into surgical scene.

## Citation

If you find this code useful for your research, please use the following BibTeX entries:

```
@misc{yang2024deform3dgs,
      title={Deform3DGS: Flexible Deformation for Fast Surgical Scene Reconstruction with Gaussian Splatting}, 
      author={Shuojue Yang and Qian Li and Daiyun Shen and Bingchen Gong and Qi Dou and Yueming Jin},
      year={2024},
      eprint={2405.17835},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
