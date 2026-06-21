from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class Volume:
    array: np.ndarray
    affine: np.ndarray
    path: Path
    spacing: tuple[float, float, float] | None = None


def read_volume(path: str | Path) -> Volume:
    image_path = Path(path)
    suffix = "".join(image_path.suffixes).lower()
    if suffix.endswith(".nii.gz") or image_path.suffix.lower() == ".nii":
        image = nib.load(str(image_path))
        return Volume(
            array=np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32),
            affine=np.asarray(image.affine, dtype=np.float32),
            path=image_path,
            spacing=tuple(float(value) for value in image.header.get_zooms()[:3]),
        )
    if image_path.suffix.lower() in {".mha", ".mhd"} or image_path.is_dir():
        return _read_simpleitk(image_path)
    raise ValueError(f"不支持的图像路径：{image_path}")


def write_volume(
    path: str | Path,
    array: np.ndarray,
    affine: np.ndarray | None = None,
    reference: Volume | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = "".join(output.suffixes).lower()
    affine_to_write = affine if affine is not None else (reference.affine if reference is not None else np.eye(4))
    if suffix.endswith(".nii.gz") or output.suffix.lower() == ".nii":
        nib.save(nib.Nifti1Image(np.asarray(array, dtype=np.float32), affine_to_write), str(output))
        return
    if output.suffix.lower() in {".mha", ".mhd"}:
        _write_simpleitk(output, array, reference)
        return
    raise ValueError(f"不支持的输出图像路径：{output}")


def _read_simpleitk(path: Path) -> Volume:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("读取 MHA/MHD 或 DICOM 输入需要安装 SimpleITK。请运行：pip install SimpleITK") from exc

    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        names = reader.GetGDCMSeriesFileNames(str(path))
        if not names:
            raise ValueError(f"目录中没有找到 DICOM 序列：{path}")
        reader.SetFileNames(names)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(path))

    array = sitk.GetArrayFromImage(image).astype(np.float32)
    array = np.transpose(array, (2, 1, 0))
    spacing = tuple(float(value) for value in image.GetSpacing())
    affine = np.diag([*spacing, 1.0]).astype(np.float32)
    return Volume(array=array, affine=affine, path=path, spacing=spacing)


def _write_simpleitk(path: Path, array: np.ndarray, reference: Volume | None) -> None:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("写出 MHA/MHD 文件需要安装 SimpleITK。请运行：pip install SimpleITK") from exc

    image = sitk.GetImageFromArray(np.transpose(np.asarray(array, dtype=np.float32), (2, 1, 0)))
    if reference and reference.spacing:
        image.SetSpacing(reference.spacing)
    sitk.WriteImage(image, str(path))
