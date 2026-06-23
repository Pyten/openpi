"""32维机器人的输入/输出转换

这个文件定义了如何将32维机器人数据转换为模型期望的格式。

32维顺序:
- 头部 2 DoF: head_pitch, head_yaw
- 左臂 7 DoF: left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow_pitch, left_wrist_yaw, left_wrist_pitch, left_wrist_roll
- 左手 6 DoF: left_little_finger, left_ring_finger, left_middle_finger, left_fore_finger, left_thumb_bend, left_thumb_rotation
- 右臂 7 DoF: right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow_pitch, right_wrist_yaw, right_wrist_pitch, right_wrist_roll
- 右手 6 DoF: right_little_finger, right_ring_finger, right_middle_finger, right_fore_finger, right_thumb_bend, right_thumb_rotation
- 腰部 2 DoF: waist_yaw, waist_pitch
- 腿部 2 DoF: hip_pitch, knee_pitch
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_grab32_example() -> dict:
    """Creates a random input example for the 32-dim robot policy."""
    return {
        "state": np.random.rand(32),  # 32维：头2 + 左臂7 + 左手6 + 右臂7 + 右手6 + 腰2 + 腿2
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "pick up the box",
    }


def _parse_image(image) -> np.ndarray:
    """解析图像为正确格式 (H, W, C) uint8"""
    image = np.asarray(image)
    # Convert to uint8 if using float images
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # Convert from [C, H, W] to [H, W, C] if needed
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Grab32Inputs(transforms.DataTransformFn):
    """将32维机器人数据转换为模型输入格式
    
    机器人配置：
    - 32维状态：头2 + 左臂7 + 左手6 + 右臂7 + 右手6 + 腰2 + 腿2
    - 32维动作：与状态相同
    - 1个相机：head camera (cam_high)
    """
    
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 解析图像（只有head camera）
        head_image = _parse_image(data["images"]["cam_high"])
        
        # 创建输入字典
        inputs = {
            "state": data["state"],  # 32维状态
            "image": {
                "base_0_rgb": head_image,  # 主相机（头部相机）
                # 填充缺失的手腕相机为零数组
                "left_wrist_0_rgb": np.zeros_like(head_image),
                "right_wrist_0_rgb": np.zeros_like(head_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # PI0-FAST需要mask=True，PI0需要mask=False表示padding
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        
        # 动作仅在训练时可用
        if "actions" in data:
            inputs["actions"] = data["actions"]  # shape: (action_horizon, 32)
        
        # 传递提示词
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        return inputs


@dataclasses.dataclass(frozen=True)
class Grab32Outputs(transforms.DataTransformFn):
    """将模型输出转换回机器人动作格式"""
    
    def __call__(self, data: dict) -> dict:
        # 返回前32维动作（如果模型padding了更多维度）
        return {"actions": np.asarray(data["actions"][:, :32])}

