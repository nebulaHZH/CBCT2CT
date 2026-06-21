# CBCT-to-CT 基线工程

这个工程实现了一个用于 SynthRAD2023 Task 2 的 CBCT-to-CT 图像转换基线。
主扩散模型是 Peng 等人官方 `conditional_DDPM` 的本地实现版本：以 CBCT 作为条件输入，以 CT 作为目标图像，用官方结构的时间嵌入 U-Net 预测扩散噪声。
对应论文为 `CBCT-Based synthetic CT image generation using conditional denoising diffusion probabilistic model`，官方仓库为 `https://github.com/junbopeng/conditional_DDPM`。

## 数据准备

先从 Zenodo 下载 SynthRAD2023，并解压 `Task2.zip`。

```powershell
python scripts/prepare_synthrad2023.py --root E:\path\to\Task2 --output-dir manifests/synthrad2023
```

期望的数据目录结构如下：

```text
Task2/
  brain/2Bxxxx/cbct.nii.gz ct.nii.gz mask.nii.gz
  pelvis/2Pxxxx/cbct.nii.gz ct.nii.gz mask.nii.gz
```

## 训练

先安装与你的 CUDA 版本匹配的 PyTorch，再安装其余依赖。

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python train.py --config configs/cddpm_synthrad2023_brain.yaml
```

如果只是想做一个极小规模的 smoke test：

```powershell
python train.py --config configs/cddpm_synthrad2023_brain.yaml --override training.epochs=1 --override data.max_slices_per_case=4
```

## 推理与评估

```powershell
python infer.py --config configs/cddpm_synthrad2023_brain.yaml --checkpoint runs/cddpm_synthrad2023_brain/checkpoints/epoch_0100.pt --manifest manifests/synthrad2023/task2_brain_train.jsonl --output-dir runs/cddpm_synthrad2023_brain/predictions
python evaluate.py --manifest manifests/synthrad2023/task2_brain_train.jsonl --prediction-dir runs/cddpm_synthrad2023_brain/predictions --output-csv runs/cddpm_synthrad2023_brain/metrics.csv --visual-dir runs/cddpm_synthrad2023_brain/visuals
```

正式比较模型前，建议先跑 identity baseline，量化“直接把原始 CBCT 当作 CT”的下限表现：

```powershell
python scripts/evaluate_identity.py --manifest manifests/synthrad2023/task2_brain_train.jsonl --output-csv runs/identity_brain_metrics.csv
```

## 建议对比的基线

- Identity baseline：直接评估原始 CBCT 与 CT 的差异，用来衡量下限。
- 官方 cDDPM baseline：默认 `configs/cddpm_synthrad2023_*.yaml`，结构对齐 `junbopeng/conditional_DDPM`，但代码在本项目中重新实现，便于适配 SynthRAD manifest。
- `configs/resunet_synthrad2023_brain.yaml`：监督式 ResUNet 回归基线。
- CycleGAN：如果后续要评估弱配对或非配对场景，可以作为经典对比方法。
- TPDM-CBCT2CT：在当前流程稳定后，可以作为更强的纹理保持扩散模型对比。

SynthRAD2023 采用 CC BY-NC 4.0 发布。除非你单独确认授权，否则实验应限于非商业研究用途。
