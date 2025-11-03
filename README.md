## 🚀 [PiDiViT: When Pixel Difference Patterns Meet ViT for Few-Shot Object Detection (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_When_Pixel_Difference_Patterns_Meet_ViT_PiDiViT_for_Few-Shot_Object_ICCV_2025_paper.pdf)

🔥**We propose PiDiViT, which empowers pretrained ViT to excel in few-shot detection by designing explicit prior modules for pixel-wise differences and multiscale variations in low-level features of pretrained ViT.**

🔥**PiDiViT achieves SOTA performance in COCO for few-shot, one-shot, and open-vocabulary object detection, setting new benchmarks and offering a valuable reference for future detecting few-shot objects.**

## 🛠️ Updates
- (11/2025) Code released with full training/evaluation scripts.
- (10/2025) Official publication in ICCV 2025 proceedings (paper: [download](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_When_Pixel_Difference_Patterns_Meet_ViT_PiDiViT_for_Few-Shot_Object_ICCV_2025_paper.pdf)).
- (06/2025) PiDiViT accepted at ICCV 2025.

## 🕸️ Dataset and Model Initialization Checkpoints
You can download them from the baseline project [DE-ViT](https://github.com/mlzxy/devit/blob/main/Downloads.md).  

## 📽️ Getting Started

### Installation
```bash
git clone https://github.com/Seaz9/PiDiViT.git
conda create -n PiDiViT python=3.9
conda activate PiDiViT
pip install -r PiDiViT/requirements.txt
pip install -e ./PiDiViT
```
### 🔍Training
```bash
vit=l task=ovd dataset=coco bash scripts/train.sh  # train open-vocabulary COCO with ViT-L

# task=ovd / fsod / osod
# dataset=coco /  voc
# vit= l 
# split = 1 / 2 / 3 / 4 for coco one shot, and 1 / 2 / 3 for voc few-shot. 

# few-shot env var `shot = 5 / 10 / 30`
vit=l task=fsod shot=10 bash scripts/train.sh 

# one-shot env var `split = 1 / 2 / 3 / 4`
vit=l task=osod split=1 bash script/train.sh

# detectron2 options can be provided through args, e.g.,
task=ovd dataset=coco bash scripts/train.sh 

# another env var is `num_gpus = 1 / 2 ...`, used to control
# how many gpus are used
```

### 🔍Evaluation

```bash
vit=l task=ovd dataset=coco bash scripts/eval.sh # evaluate COCO OVD with ViT-L/14

# evaluate Pascal VOC split-3 with ViT-L/14 with 5 shot
vit=l task=fsod dataset=voc split=3 shot=5 bash scripts/eval.sh 
```


Check [Tools.md](Tools.md) for intructions to build prototype and prepare weights (for your custom datasets).

📜 Citation
```
@InProceedings{Zhou_2025_ICCV,
    author    = {Zhou, Hongliang and Liu, Yongxiang and Mo, Canyu and Li, Weijie and Peng, Bowen and Liu, Li},
    title     = {When Pixel Difference Patterns Meet ViT: PiDiViT for Few-Shot Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {24309-24318}
}
```
📜 License

This project is under the Apache-2.0 license.

⚙️ Acknowledgement

PiDiViT builds upon the good work of [DE-ViT](https://github.com/mlzxy/devit). Special thanks to the DE-ViT team for their exceptional open-source contributions.

⭐ Support the Project

If PiDiViT accelerates your research, please ⭐ the repository and cite it to support future development.

Your stars fuel the next breakthrough in few-shot detection! 🔥🚀
