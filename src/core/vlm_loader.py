from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


def load_vlm_model(model_id: str, **kwargs: Any):
    """
    Load a vision-language generation model with broad Transformers compatibility.
    Prefer auto classes so we can switch between Qwen2.5-VL and Qwen3-VL cleanly.

    When CUDA is available, defaults to placing the **entire** model on GPU
    ``cuda:VLM_CUDA_DEVICE`` (device_map ``{\"\": idx}``), avoiding ``device_map=\"auto\"``
    silently offloading layers to CPU (which makes inference very slow).

    Override with env ``VLM_DEVICE_MAP=auto`` if you need accelerate auto/sharding.

    INT8（GPU）：设置 ``VLM_LOAD_IN_8BIT=true`` 时使用 ``bitsandbytes`` 的 ``load_in_8bit``，
    权重量化显存通常更低（视觉分支是否量化取决于 Transformers 实现）。
    """
    kw = dict(kwargs)
    if "torch_dtype" in kw and "dtype" not in kw:
        kw["dtype"] = kw.pop("torch_dtype")

    cuda_available = torch.cuda.is_available()
    force_cuda = os.getenv("VLM_FORCE_CUDA", "true").lower() in {"1", "true", "yes", "on"}
    dm_env = os.getenv("VLM_DEVICE_MAP", "").strip()
    use_8bit = os.getenv("VLM_LOAD_IN_8BIT", "").lower() in {"1", "true", "yes", "on"}

    if use_8bit:
        if not cuda_available:
            raise RuntimeError("VLM_LOAD_IN_8BIT=true 需要 CUDA；当前未检测到 GPU。")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("INT8 推理需要安装 bitsandbytes：pip install bitsandbytes") from exc
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kw.pop("dtype", None)
        kw.pop("torch_dtype", None)
        logger.info("load_vlm_model | quantization=int8(bitsandbytes)")
        if dm_env.lower() == "auto":
            kw["device_map"] = "auto"
        elif dm_env:
            kw["device_map"] = dm_env
        else:
            idx = int(os.getenv("VLM_CUDA_DEVICE", "0"))
            kw["device_map"] = {"": idx}
        gpu_name = ""
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown"
        logger.info(
            "load_vlm_model | device_map=%s | visible_cuda:0=%s",
            kw.get("device_map"),
            gpu_name,
        )
    elif cuda_available and force_cuda:
        if dm_env.lower() == "auto":
            kw["device_map"] = "auto"
        elif dm_env:
            kw["device_map"] = dm_env
        else:
            idx = int(os.getenv("VLM_CUDA_DEVICE", "0"))
            kw["device_map"] = {"": idx}
        gpu_name = ""
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "unknown"
        logger.info(
            "load_vlm_model | device_map=%s | visible_cuda:0=%s",
            kw.get("device_map"),
            gpu_name,
        )
    else:
        if cuda_available and not force_cuda:
            kw.setdefault("device_map", "cpu")
            logger.warning("VLM_FORCE_CUDA=false：强制在 CPU 上加载（很慢）。")
        else:
            kw.setdefault("device_map", "cpu")
            logger.warning("未检测到 CUDA：在 CPU 上加载 VLM（会很慢）。请检查驱动与 PyTorch CUDA 版本。")

    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    except Exception:
        try:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq.from_pretrained(model_id, **kw)
        except Exception:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kw)
