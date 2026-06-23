#!/usr/bin/env python3
"""
OpenPI ROS2 客户端（离线部署示例）

功能：
- 订阅机器人相关的 ROS2 topics（相机图像 + 电机/手部状态）
- 组装为 OpenPI 所需的观测格式，通过 WebSocket 连接策略服务器进行推理
- 将推理得到的动作结果发布为 ROS2 topic，供下游控制模块使用

参考：
- 自定义抓取机器人客户端：old_client.py（状态/动作维度定义、OpenPI 调用方式）
- OpenPI 官方离线客户端示例：offline_client.py（image_tools + WebsocketClientPolicy 用法）
- ROS2 订阅与 socket 客户端示例：robot_client_ros2.py（topic 定义与数据结构）
"""

import os
import time
import json
import logging
import math
import pdb
import bisect
import atexit
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from openpi_client import image_tools
from openpi_client import websocket_client_policy

# 在导入 rclpy 之前控制 ROS2 日志级别，避免过多输出
if "RCUTILS_LOGGING_SEVERITY" not in os.environ:
    os.environ["RCUTILS_LOGGING_SEVERITY"] = "WARN"

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float32MultiArray
    import cv2
    from cv_bridge import CvBridge

    # 尝试导入与 robot_client_ros2.py 一致的自定义电机状态消息
    try:
        from bodyctrl_msgs.msg import MotorStatusMsg, CmdSetMotorPosition, SetMotorPosition  # type: ignore

        CUSTOM_MSGS_AVAILABLE = True
    except ImportError:
        MotorStatusMsg = None  # type: ignore
        CmdSetMotorPosition = None  # type: ignore
        SetMotorPosition = None  # type: ignore
        CUSTOM_MSGS_AVAILABLE = False

    ROS2_AVAILABLE = True
except ImportError as e:  # pragma: no cover - 环境问题
    print(f"[openpi_ros2_client] ROS2 相关依赖导入失败: {e}")
    ROS2_AVAILABLE = False
    CUSTOM_MSGS_AVAILABLE = False


log_level = os.environ.get("ROBOT_CLIENT_LOG_LEVEL", "INFO").upper()
log_file = os.environ.get("ROBOT_CLIENT_LOG_FILE", None)

# 配置日志：同时输出到控制台和文件（如果指定）
handlers = []
# 控制台输出
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
handlers.append(console_handler)

# 文件输出（如果指定了日志文件）
if log_file:
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    handlers.append(file_handler)

logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    handlers=handlers
)
logger = logging.getLogger("openpi_ros2_client")


def _env_truthy(name: str, default: str = "false") -> bool:
    """True for 1 / true / yes / on (case-insensitive), matching typical shell exports like SAVE_ACTIONS=1."""
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


if log_file:
    logger.info(f"日志同时输出到文件: {log_file}")


class OpenPIRos2Client(Node):
    """
    OpenPI ROS2 客户端节点

    - 状态维度: 32 (头部2维 + 原26维 + 腰腿4维)
    - 动作维度: 32 (头部2维 + 原26维 + 腰腿4维)
    - 相机: 
      - 头部相机: `/camera/color/image_raw`
      - 左腕部相机: `/D405/left/color/image_raw`
      - 右腕部相机: `/D405/right/color/image_raw`

    订阅（对齐 robot_client_ros2.py）：
    - 图像: 
      - `/camera/color/image_raw` (sensor_msgs/Image) - 头部相机
      - `/D405/left/color/image_raw` (sensor_msgs/Image) - 左腕部相机
      - `/D405/right/color/image_raw` (sensor_msgs/Image) - 右腕部相机
    - 手部状态: `/inspire_hand/state/left_hand`, `/inspire_hand/state/right_hand` (sensor_msgs/JointState)
    - 电机状态: `/head/status`, `/waist/status`, `/arm/status`, `/leg/status` (bodyctrl_msgs/MotorStatusMsg，可选)

    发布：
    - 头部命令: `/head/cmd_pos` (bodyctrl_msgs/CmdSetMotorPosition)
    - 臂部命令: `/arm/cmd_pos` (bodyctrl_msgs/CmdSetMotorPosition)
    - 手部命令: `/inspire_hand/ctrl/left_hand`, `/inspire_hand/ctrl/right_hand` (sensor_msgs/JointState)
    - 腰部命令: `/waist/cmd_pos` (bodyctrl_msgs/CmdSetMotorPosition)
    - 腿部命令: `/leg/cmd_pos` (bodyctrl_msgs/CmdSetMotorPosition)
    """

    def __init__(
        self,
        policy_host: str = "localhost",
        policy_port: int = 8000,
        control_frequency: float = 10.0,
        action_horizon: int = 10,
        prompt: str = "pick up the box",
    ):
        super().__init__("openpi_ros2_client")

        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS2 不可用，无法启动 OpenPIRos2Client")

        # OpenPI/策略服务器配置
        self.policy_host = policy_host
        self.policy_port = policy_port
        self.control_frequency = control_frequency
        self.control_period = 1.0 / control_frequency
        self.action_horizon = action_horizon
        self.prompt = prompt

        # 机器人配置（32维：头部2维 + 原26维 + 腰腿4维）
        self.state_dim = 32
        self.action_dim = 32
        self.image_height = 224
        self.image_width = 224
        
        # 关节角度限制（根据 demension.json，角度范围转换为弧度）
        # 格式：{name: (min_rad, max_rad)}
        # 头部关节 (name: 2-3)
        self.joint_limits = {
            2: (-26 * math.pi / 180, 26 * math.pi / 180),     # head_pitch: 【-26， 26】度
            3: (-25 * math.pi / 180, 25 * math.pi / 180),     # head_yaw: 【-25， 25】度
            # 左臂关节 (name: 11-17)
            11: (-170 * math.pi / 180, 170 * math.pi / 180),  # left_shoulder_pitch: 【-170， 170】度
            12: (-15 * math.pi / 180, 150 * math.pi / 180),   # left_shoulder_roll: 【-15， 150】度
            13: (-170 * math.pi / 180, 170 * math.pi / 180),  # left_shoulder_yaw: 【-170， 170】度
            14: (-150 * math.pi / 180, 15 * math.pi / 180),   # left_elbow_pitch: 【-150， 15】度
            15: (-170 * math.pi / 180, 170 * math.pi / 180),  # left_wrist_yaw: 【-170， 170】度
            16: (-45 * math.pi / 180, 60 * math.pi / 180),     # left_wrist_pitch: 【-45， 60】度
            17: (-95 * math.pi / 180, 75 * math.pi / 180),     # left_wrist_roll: 【-95， 75】度
            # 右臂关节 (name: 21-27)
            21: (-170 * math.pi / 180, 170 * math.pi / 180),  # right_shoulder_pitch: 【-170， 170】度
            22: (-150 * math.pi / 180, 15 * math.pi / 180),   # right_shoulder_roll: 【-150， 15】度
            23: (-170 * math.pi / 180, 170 * math.pi / 180),  # right_shoulder_yaw: 【-170， 170】度
            24: (-150 * math.pi / 180, 15 * math.pi / 180),  # right_elbow_pitch: 【-150， 15】度
            25: (-170 * math.pi / 180, 170 * math.pi / 180),  # right_wrist_yaw: 【-170， 170】度
            26: (-45 * math.pi / 180, 60 * math.pi / 180),    # right_wrist_pitch: 【-45， 60】度
            27: (-75 * math.pi / 180, 95 * math.pi / 180),    # right_wrist_roll: 【-75， 95】度
            # 腰部关节 (name: 31-32)
            31: (-160 * math.pi / 180, 180 * math.pi / 180),  # waist_yaw: 【-160， 180】度
            32: (-45 * math.pi / 180, 120 * math.pi / 180),   # waist_pitch: 【-45， 120】度
            # 腿部关节 (name: 51-52)
            51: (13 * math.pi / 180, 80 * math.pi / 180),     # hip_pitch: 【13， 80】度
            52: (26 * math.pi / 180, 160 * math.pi / 180),    # knee_pitch: 【26， 160】度
        }
        
        # 手部关节角度限制（根据 demension.json，角度范围转换为弧度）
        # 格式：{name: (min_rad, max_rad)}，name 为字符串 '1'-'6'
        # 左手和右手使用相同的限制
        self.hand_joint_limits = {
            '1': (19 * math.pi / 180, 176 * math.pi / 180),   # little_finger: 【19， 176】度
            '2': (19 * math.pi / 180, 176 * math.pi / 180),   # ring_finger: 【19， 176】度
            '3': (19 * math.pi / 180, 176 * math.pi / 180),   # middle_finger: 【19， 176】度
            '4': (19 * math.pi / 180, 176 * math.pi / 180),   # fore_finger: 【19， 176】度
            '5': (-13 * math.pi / 180, 53 * math.pi / 180),   # thumb_bend: 【-13， 53】度
            '6': (90 * math.pi / 180, 165 * math.pi / 180),   # thumb_rotation: 【90， 165】度
        }

        # OpenPI 策略客户端
        self.policy_client: Optional[websocket_client_policy.WebsocketClientPolicy] = None

        # ROS2 相关
        self.bridge = CvBridge()

        # 事件触发模式：缓存数据和时间戳
        # 图像数据（带时间戳）
        # 头部相机
        self.latest_image: Optional[np.ndarray] = None
        self.latest_image_timestamp: Optional[float] = None
        # 腕部相机（左）
        self.latest_left_wrist_image: Optional[np.ndarray] = None
        self.latest_left_wrist_image_timestamp: Optional[float] = None
        # 腕部相机（右）
        self.latest_right_wrist_image: Optional[np.ndarray] = None
        self.latest_right_wrist_image_timestamp: Optional[float] = None
        
        # 手部状态（带时间戳）
        self.latest_left_hand: Optional[JointState] = None
        self.latest_left_hand_timestamp: Optional[float] = None
        self.latest_right_hand: Optional[JointState] = None
        self.latest_right_hand_timestamp: Optional[float] = None
        
        # 电机状态（带时间戳，与 robot_client_ros2 对齐，主要使用 /arm/status）
        self.latest_head_status: Optional[Any] = None
        self.latest_head_status_timestamp: Optional[float] = None
        self.latest_waist_status: Optional[Any] = None
        self.latest_waist_status_timestamp: Optional[float] = None
        self.latest_arm_status: Optional[Any] = None
        self.latest_arm_status_timestamp: Optional[float] = None
        self.latest_leg_status: Optional[Any] = None
        self.latest_leg_status_timestamp: Optional[float] = None
        
        # 图像处理缓存（避免重复处理相同图像，降低CPU占用）
        self.cached_processed_image: Optional[np.ndarray] = None
        self.cached_image_id: Optional[int] = None  # 使用图像对象的id作为缓存键
        # 腕部相机图像处理缓存
        self.cached_processed_left_wrist_image: Optional[np.ndarray] = None
        self.cached_left_wrist_image_id: Optional[int] = None
        self.cached_processed_right_wrist_image: Optional[np.ndarray] = None
        self.cached_right_wrist_image_id: Optional[int] = None
        
        # 时间同步容差（秒）：允许status数据与图像的最大时间差
        # 由于status数据可能更新频率较低，增大容差到500ms
        self.status_time_tolerance = 0.5  # 500ms
        
        # 历史数据缓存（保存3秒的数据）
        self.cache_history_seconds = 3.0
        self.cache_max_size = 100  # 每个topic最大缓存条目数，防止内存溢出
        self.data_cache = {
            'image': [],                    # [(ros2_timestamp, receive_time, data), ...]
            'left_wrist_image': [],         # [(ros2_timestamp, receive_time, data), ...]
            'right_wrist_image': [],       # [(ros2_timestamp, receive_time, data), ...]
            'left_hand': [],                # [(ros2_timestamp, receive_time, data), ...]
            'right_hand': [],               # [(ros2_timestamp, receive_time, data), ...]
            'head_status': [],              # [(ros2_timestamp, receive_time, data), ...]
            'waist_status': [],              # [(ros2_timestamp, receive_time, data), ...]
            'arm_status': [],                # [(ros2_timestamp, receive_time, data), ...]
            'leg_status': [],                # [(ros2_timestamp, receive_time, data), ...]
        }
        # 缓存统计信息
        self.cache_stats = {
            'image': {'hits': 0, 'misses': 0, 'evictions': 0},
            'left_wrist_image': {'hits': 0, 'misses': 0, 'evictions': 0},
            'right_wrist_image': {'hits': 0, 'misses': 0, 'evictions': 0},
            'left_hand': {'hits': 0, 'misses': 0, 'evictions': 0},
            'right_hand': {'hits': 0, 'misses': 0, 'evictions': 0},
            'head_status': {'hits': 0, 'misses': 0, 'evictions': 0},
            'waist_status': {'hits': 0, 'misses': 0, 'evictions': 0},
            'arm_status': {'hits': 0, 'misses': 0, 'evictions': 0},
            'leg_status': {'hits': 0, 'misses': 0, 'evictions': 0},
        }
        
        # 初始值配置（首次跳过推理，直接发布初始值）
        # self.initial_action = np.array([
        #     0.362901, -0.034553, 0.271297, 0.108702, 0.448554, -1.256885, -0.002716, -0.493727,
        #     -0.000817, 0.989490, 1.000000, 1.000000, 1.000000, 0.991933, 0.981017, 0.174192,
        #     -0.264507, -0.405815, -1.182983, 0.218145, -0.508831, 0.054037, 0.998166, 0.997730,
        #     0.996253, 0.996424, 0.994756, 0.00, 0.00, 0.00, 0.00, 0.5
        # ], dtype=np.float32)

        self.initial_action_1 = np.array([
            0.44339895248413086,
            -0.005561351776123047,
           0.1645197868347168,
            0.051723480224609375,
            -0.2327098846435547,
            -0.22417736053466797,
            0.19205951690673828,
            -0.12937545776367188,
            -0.020513057708740234,
            0.9919999837875366,
            1.0,
            1.0,
            1.0,
            0.9950000047683716,
            0.9829999804496765,
           0.21535634994506836,
            -0.045780181884765625,
            0.06212663650512695,
            -0.3538703918457031,
            -0.11399412155151367,
            0.011979103088378906,
            0.02258014678955078,
            1.0,
            0.9959999918937683,
            0.9980000257492065,
            0.9980000257492065,
            0.996999979019165,
            0.0,
            -0.0028522454667836428,
            -4.793690095539205e-05,
            0.0005992112564854324,
            0.4995743930339813
        ], dtype=np.float32)

        self.initial_action_2 = np.array([
            0.44339895248413086,
            -0.005561351776123047,
           0.1645197868347168,
            1.51723480224609375,
            -0.2327098846435547,
            -0.22417736053466797,
            0.19205951690673828,
            -0.12937545776367188,
            -0.020513057708740234,
            0.9919999837875366,
            1.0,
            1.0,
            1.0,
            0.9950000047683716,
            0.9829999804496765,
           0.21535634994506836,
            -1.545780181884765625,
            0.06212663650512695,
            -0.3538703918457031,
            -0.11399412155151367,
            0.011979103088378906,
            0.02258014678955078,
            1.0,
            0.9959999918937683,
            0.9980000257492065,
            0.9980000257492065,
            0.996999979019165,
            0.0,
            -0.0028522454667836428,
            -4.793690095539205e-05,
            0.0005992112564854324,
            0.4995743930339813
        ], dtype=np.float32)

        self.initial_action_3 = np.array([
            0.44339895248413086,
            -0.005561351776123047,
            -0.47191476821899414,
            0.0032830238342285156,
            -0.2332615852355957,
            -0.7047443389892578,
            0.19797945022583008,
            -0.37505239248275757,
            -0.023496977984905243,
            0.9919999837875366,
            1.0,
            1.0,
            1.0,
            0.9950000047683716,
            0.9829999804496765,
            -0.3012833595275879,
            -0.002444744110107422,
            0.09635305404663086,
            -1.0180363655090332,
            -0.00023984909057617188,
            -0.1608271598815918,
            0.018964767456054688,
            1.0,
            0.9959999918937683,
            0.9980000257492065,
            0.9980000257492065,
            0.996999979019165,
            0.0,
            -0.0028522454667836428,
            -4.793690095539205e-05,
            0.0005992112564854324,
            0.4995743930339813
        ], dtype=np.float32)

        self.putdown_initial_action_1 = np.array([
            0.44339895248413086,
            -0.005561351776123047,
            -1.2063078880310059,
            -4.76837158203125e-07,
            -0.20502614974975586,
            -0.8157901763916016,
            0.17072725296020508,
            0.2829289436340332,
            -0.10512208938598633,
            0.9919999837875366,
            1.0,
            1.0,
            1.0,
            0.9950000047683716,
            0.9829999804496765,
            -0.9288253784179688,
            -4.76837158203125e-07,
            0.11500120162963867,
            -1.2412781715393066,
            -0.15447664260864258,
            0.06418609619140625,
            0.0931253433227539,
                        1.0,
            0.9959999918937683,
            0.9980000257492065,
            0.9980000257492065,
            0.996999979019165,
            0.0,
            3.14,
            -4.793690095539205e-05,
            0.0005992112564854324,
            0.4995743930339813
            ], dtype=np.float32)




        self.putdown_initial_action_2 = np.array([
            0.44339895248413086,
            -0.005561351776123047,
            -1.2063078880310059,
            -4.76837158203125e-07,
            -0.20502614974975586,
            -0.8157901763916016,
            0.17072725296020508,
            0.2829289436340332,
            -0.10512208938598633,
            0.9919999837875366,
            1.0,
            1.0,
            1.0,
            0.9950000047683716,
            0.9829999804496765,
            -0.9288253784179688,
            -4.76837158203125e-07,
            0.11500120162963867,
            -1.2412781715393066,
            -0.15447664260864258,
            0.06418609619140625,
            0.0931253433227539,
                        1.0,
            0.9959999918937683,
            0.9980000257492065,
            0.9980000257492065,
            0.996999979019165,
            0.0,
            0.0,
            -4.793690095539205e-05,
            0.0005992112564854324,
            0.4995743930339813
            ], dtype=np.float32)

        self.initial_action_published = False  # 标记是否已发布初始值
        
        # 推理结果保存配置（SAVE_ACTIONS=1 与 true 均有效，与 run_pick_put_sequence_dual_model.sh 一致）
        self.save_actions = _env_truthy("SAVE_ACTIONS", "false")
        self.actions_save_dir = os.environ.get("ACTIONS_SAVE_DIR", "/tmp/robot_actions")
        self.action_save_counter = 0  # 推理结果计数器
        self._session_npy_list: List[np.ndarray] = []  # 当前会话累积的 actions
        self._session_state_list: List[np.ndarray] = []  # 当前会话累积的 states
        self._session_meta_list: List[Tuple[float, int]] = []  # (infer_time, horizon) per inference
        self._session_infer_times: List[float] = []  # 各次推理耗时
        self._atexit_registered = False
        if self.save_actions:
            atexit.register(self._finalize_actions_session)
        if self.save_actions:
            os.makedirs(self.actions_save_dir, exist_ok=True)
            logger.info(f"推理结果保存已启用，保存目录: {self.actions_save_dir}")

        # 是否在控制台/日志中打印每次推理的完整动作序列（设 PRINT_INFERENCE_ACTIONS=1）
        self.print_inference_actions = _env_truthy("PRINT_INFERENCE_ACTIONS", "false")
        if self.print_inference_actions:
            logger.info("已启用 PRINT_INFERENCE_ACTIONS：每次推理会在日志中输出完整 actions 矩阵")

        # 推理原始结果（JSON）保存配置
        self.save_infer_result = _env_truthy("SAVE_INFER_RESULT", "false")
        self.infer_result_save_dir = os.environ.get("INFER_RESULT_SAVE_DIR", "/home/ubuntu/robot_communication/json_output")
        self.infer_result_counter = 0
        if self.save_infer_result:
            os.makedirs(self.infer_result_save_dir, exist_ok=True)
            logger.info(f"推理原始结果保存已启用，保存目录: {self.infer_result_save_dir}")

        # 每次推理将动作序列写入 JSON（策略输出；下发前还会在 _publish_action_vector 做限位）
        self.save_action_information = _env_truthy("SAVE_ACTION_INFORMATION", "true")
        self.action_information_dir = os.environ.get(
            "ACTION_INFORMATION_DIR", "/home/ubuntu/robot_communication/action_information"
        )
        self.action_information_counter = 0
        if self.save_action_information:
            os.makedirs(self.action_information_dir, exist_ok=True)
            logger.info(
                "ACTION_INFORMATION JSON 已启用，目录: %s（关闭: SAVE_ACTION_INFORMATION=0）",
                self.action_information_dir,
            )

        # 图像保存配置（只保存一帧，写死启用）
        self.save_images = True  # 写死为启用
        self.image_save_dir = os.getcwd()  # 保存到脚本运行的当前工作目录
        self.image_saved = False  # 标记是否已保存头部相机图像（只保存一帧）
        self.left_wrist_image_saved = False  # 标记是否已保存左腕部相机图像（只保存一帧）
        self.right_wrist_image_saved = False  # 标记是否已保存右腕部相机图像（只保存一帧）
        os.makedirs(self.image_save_dir, exist_ok=True)
        logger.info(f"图像保存已启用（只保存一帧），保存目录: {self.image_save_dir}")

        # 发布推理动作的 topics（按照 demension.json 的映射）
        # 头部2个关节 -> /head/cmd_pos (CmdSetMotorPosition)
        if CUSTOM_MSGS_AVAILABLE and CmdSetMotorPosition is not None:
            self.head_cmd_pub = self.create_publisher(
                CmdSetMotorPosition, "/head/cmd_pos", 10
            )
            # 左臂7个关节 + 右臂7个关节 -> /arm/cmd_pos (CmdSetMotorPosition)
            self.arm_cmd_pub = self.create_publisher(
                CmdSetMotorPosition, "/arm/cmd_pos", 10
            )
            # 腰部2个关节 -> /waist/cmd_pos (CmdSetMotorPosition)
            self.waist_cmd_pub = self.create_publisher(
                CmdSetMotorPosition, "/waist/cmd_pos", 10
            )
            # 腿部2个关节 -> /leg/cmd_pos (CmdSetMotorPosition)
            self.leg_cmd_pub = self.create_publisher(
                CmdSetMotorPosition, "/leg/cmd_pos", 10
            )
        else:
            self.head_cmd_pub = None
            self.arm_cmd_pub = None
            self.waist_cmd_pub = None
            self.leg_cmd_pub = None
            logger.warning("CmdSetMotorPosition 不可用，无法发布命令 topics")
        
        # 左手6个手指 -> /inspire_hand/ctrl/left_hand (JointState)
        self.left_hand_cmd_pub = self.create_publisher(
            JointState, "/inspire_hand/ctrl/left_hand", 10
        )
        
        # 右手6个手指 -> /inspire_hand/ctrl/right_hand (JointState)
        self.right_hand_cmd_pub = self.create_publisher(
            JointState, "/inspire_hand/ctrl/right_hand", 10
        )

        # 订阅 ROS2 topics（参考 robot_client_ros2.py 中的定义）
        self._create_subscriptions()

        # 初始化 OpenPI 策略客户端（参考 old_client.py + offline_client.py）
        self._initialize_policy_client()

        # 定时器触发模式：每5秒触发一次推理检查
        timer_period = 0.5 # 5秒
        self.timer = self.create_timer(timer_period, self.timer_callback)
        logger.info(f"使用定时器触发模式：每{timer_period}秒触发一次推理检查")

        logger.info(
            f"OpenPIRos2Client 初始化完成（定时器触发模式）, policy server: {self.policy_host}:{self.policy_port}, "
            f"prompt='{self.prompt}', 时间同步容差={self.status_time_tolerance*1000:.0f}ms, 定时器周期={timer_period}s"
        )

    # --------------------------------------------------------------------- #
    # ROS2 topic 订阅
    # --------------------------------------------------------------------- #
    def _create_subscriptions(self):
        """创建所有需要的订阅"""
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

        # 头部相机使用BEST_EFFORT
        qos_profile = QoSProfile(
            depth=500,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        
        # 腕部相机使用RELIABLE + TRANSIENT_LOCAL（匹配发布者的QoS设置）
        # 发布者QoS: Reliability=RELIABLE, Durability=TRANSIENT_LOCAL, History (Depth): UNKNOWN
        # 系统默认QoS是BEST_EFFORT+VOLATILE，与发布者不匹配，必须显式设置TRANSIENT_LOCAL
        # TRANSIENT_LOCAL需要depth参数，发布者显示UNKNOWN，尝试使用常见的depth值
        qos_profile_reliable = QoSProfile(
            depth=10,  # 尝试使用depth=10（常见的默认值）
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # 必须匹配发布者的TRANSIENT_LOCAL
            history=HistoryPolicy.KEEP_LAST,
        )
        logger.info(f"[订阅] 腕部相机QoS配置: reliability=RELIABLE, durability=TRANSIENT_LOCAL, depth=10")
        
        start_time = time.time()
        # 相机图像（头部相机）
        logger.info(f"[头部相机] 订阅被触发")
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            qos_profile,
        )
        # 腕部相机（左）- 使用RELIABLE + TRANSIENT_LOCAL QoS
        logger.info(f"[订阅] 开始订阅左腕部相机: /D405/left/color/image_raw (QoS: RELIABLE + TRANSIENT_LOCAL)")
        left_wrist_sub = self.create_subscription(
            Image,
            "/D405/left/color/image_raw",
            self.left_wrist_image_callback,
            qos_profile_reliable,  # 使用RELIABLE + TRANSIENT_LOCAL QoS
        )
        logger.info(f"[订阅] 左腕部相机订阅对象已创建: {left_wrist_sub}")
        
        # 腕部相机（右）- 使用RELIABLE QoS
        logger.info(f"[订阅] 开始订阅右腕部相机: /D405/right/color/image_raw (使用RELIABLE QoS)")
        right_wrist_sub = self.create_subscription(
            Image,
            "/D405/right/color/image_raw",
            self.right_wrist_image_callback,
            qos_profile_reliable,  # 使用RELIABLE QoS
        )
        logger.info(f"[订阅] 右腕部相机订阅对象已创建: {right_wrist_sub}")
        end_time = time.time()
        subscription_time = (end_time - start_time) * 1000  # 转换为毫秒
        logger.info(f"[订阅] 订阅时间: {subscription_time:.2f}ms")
        # 手部状态 JointState（左手）
        self.create_subscription(
            JointState,
            "/inspire_hand/state/left_hand",
            self.left_hand_callback,
            qos_profile,
        )

        # 手部状态 JointState（右手）
        self.create_subscription(
            JointState,
            "/inspire_hand/state/right_hand",
            self.right_hand_callback,
            qos_profile,
        )

        # 电机状态（只有在自定义消息可用时才订阅）
        if CUSTOM_MSGS_AVAILABLE and MotorStatusMsg is not None:
            # 与 robot_client_ros2.py 中 topics 一致
            self.create_subscription(
                MotorStatusMsg,
                "/head/status",
                self._make_motor_status_callback("head"),
                qos_profile,
            )
            self.create_subscription(
                MotorStatusMsg,
                "/waist/status",
                self._make_motor_status_callback("waist"),
                qos_profile,
            )
            self.create_subscription(
                MotorStatusMsg,
                "/arm/status",
                self._make_motor_status_callback("arm"),
                qos_profile,
            )
            self.create_subscription(
                MotorStatusMsg,
                "/leg/status",
                self._make_motor_status_callback("leg"),
                qos_profile,
            )
            logger.info(
                "已订阅: /camera/color/image_raw, /D405/left/color/image_raw, /D405/right/color/image_raw, "
                "/inspire_hand/state/left_hand, /inspire_hand/state/right_hand, "
                "/head/status, /waist/status, /arm/status, /leg/status"
            )
        else:
            logger.warning(
                "未能导入 bodyctrl_msgs/MotorStatusMsg，仅订阅手部和相机 topic，"
                "状态向量中的 14 维臂部关节将暂时置零"
            )
            logger.info(
                "已订阅: /camera/color/image_raw, /D405/left/color/image_raw, /D405/right/color/image_raw, "
                "/inspire_hand/state/left_hand, /inspire_hand/state/right_hand"
            )

    def _extract_ros2_timestamp(self, msg) -> float:
        """
        从ROS2消息中提取时间戳（秒）。
        使用header.stamp，如果消息没有header或stamp则使用当前系统时间作为fallback。
        """
        try:
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp = msg.header.stamp
                if hasattr(stamp, 'sec') and hasattr(stamp, 'nanosec'):
                    # ROS2时间戳：sec + nanosec / 1e9
                    # 即使sec=0和nanosec=0也使用（可能是未设置的时间戳，但仍然是ROS2时间基准）
                    return float(stamp.sec) + float(stamp.nanosec) / 1e9
        except Exception as e:
            logger.debug(f"提取ROS2时间戳失败: {e}，使用系统时间作为fallback")
        
        # Fallback: 如果消息没有header或stamp，使用系统时间
        return time.time()

    def _cache_data(self, cache_key: str, timestamp: float, data: Any):
        """
        优化的缓存管理函数：将数据存入缓存，智能清理历史数据
        
        优化点：
        1. 同时考虑接收时间和ROS2时间戳进行清理
        2. 限制缓存大小，防止内存溢出
        3. 保持缓存按ROS2时间戳有序（便于快速查找）
        
        Args:
            cache_key: 缓存键名 ('image', 'left_wrist_image', 'right_wrist_image', 'left_hand', 'right_hand', 'head_status', 'waist_status', 'arm_status', 'leg_status')
            timestamp: 数据时间戳（ROS2时间戳）
            data: 要缓存的数据
        """
        if cache_key not in self.data_cache:
            return
        
        receive_time = time.time()  # 接收时间（系统时间）
        
        # 添加新数据：格式为 (ros2_timestamp, receive_time, data)
        # 保持按ROS2时间戳有序插入（使用bisect优化）
        cache_list = self.data_cache[cache_key]
        
        # 如果缓存为空或新数据时间戳最大，直接追加（常见情况，O(1)）
        if not cache_list or timestamp >= cache_list[-1][0]:
            cache_list.append((timestamp, receive_time, data))
        else:
            # 否则找到插入位置（保持有序，O(log n)）
            insert_pos = bisect.bisect_left([item[0] for item in cache_list], timestamp)
            cache_list.insert(insert_pos, (timestamp, receive_time, data))
        
        # 智能清理策略：
        # 1. 找到所有topic最新的ROS2时间戳（每个topic的最后一帧）
        # 2. 在这些最新时间戳中，找到最老的那个
        # 3. 基于这个最老的最新时间戳计算cutoff，保留从cutoff到各topic最新时间戳的数据
        # 4. 限制缓存大小（防止内存溢出）
        
        # 找到所有topic最新的ROS2时间戳，然后在这些最新时间戳中找到最老的
        # 这样保证所有topic都保留从(oldest_latest_ts - cache_history_seconds)到各自最新时间戳的数据
        latest_ros2_timestamps = []
        for topic_key, topic_cache in self.data_cache.items():
            if topic_cache:  # 如果该topic有数据
                topic_latest_ts = topic_cache[-1][0]  # 最后一个元素是最新的ROS2时间戳（已排序）
                latest_ros2_timestamps.append(topic_latest_ts)
        
        # 在所有最新时间戳中找到最老的那个
        if latest_ros2_timestamps:
            oldest_latest_ts = min(latest_ros2_timestamps)
            # 基于这个最老的最新时间戳计算cutoff
            # cutoff = oldest_latest_ts - cache_history_seconds
            # 例如：oldest_latest_ts=98.0, cache_history_seconds=3.0, 则cutoff=95.0
            # 这样所有topic都保留从95.0到各自最新时间戳的数据
            # image保留95.0-100.0, left_hand保留95.0-98.0, arm_status保留95.0-99.0
            ros2_cutoff = oldest_latest_ts - self.cache_history_seconds
        else:
            # 如果没有其他topic的数据，使用当前topic的最新ROS2时间戳
            if cache_list:
                latest_ros2_ts = cache_list[-1][0]
                ros2_cutoff = latest_ros2_ts - self.cache_history_seconds
            else:
                ros2_cutoff = float('-inf')
        
        # 统一清理策略：基于ROS2时间戳删除历史数据
        # 只保留ROS2时间戳 >= ros2_cutoff 的数据
        # 这样所有topic都保留从cutoff到各自最新时间戳的数据（例如：95.0到100.0范围内的数据）
        original_size = len(cache_list)
        self.data_cache[cache_key] = [
            (ts, rt, d) for ts, rt, d in cache_list
            if ts >= ros2_cutoff
        ]
        
        # 如果缓存仍然太大，删除最旧的数据（基于ROS2时间戳）
        if len(self.data_cache[cache_key]) > self.cache_max_size:
            # 缓存已按ROS2时间戳排序，直接保留最新的cache_max_size条
            self.data_cache[cache_key] = self.data_cache[cache_key][-self.cache_max_size:]
            evicted = original_size - len(self.data_cache[cache_key])
            if cache_key in self.cache_stats:
                self.cache_stats[cache_key]['evictions'] += evicted
    
    def _get_latest_from_cache(self, cache_key: str) -> Optional[tuple]:
        """
        从缓存中获取最新的数据
        
        Args:
            cache_key: 缓存键名
            
        Returns:
            (timestamp, data) 或 None
        """
        if cache_key not in self.data_cache or not self.data_cache[cache_key]:
            if cache_key in self.cache_stats:
                self.cache_stats[cache_key]['misses'] += 1
            return None
        # 返回格式：((ros2_timestamp, receive_time, data) -> (ros2_timestamp, data))
        ros2_ts, receive_ts, data = self.data_cache[cache_key][-1]
        if cache_key in self.cache_stats:
            self.cache_stats[cache_key]['hits'] += 1
        return (ros2_ts, data)
    
    def _get_data_by_timestamp(self, cache_key: str, target_timestamp: float, max_time_diff: float = 2.0) -> Optional[tuple]:
        """
        优化的时间戳查找：使用二分查找快速定位最接近的数据（时间同步用）
        
        优化点：
        1. 使用二分查找，O(log n)复杂度（原来O(n)）
        2. 缓存已按ROS2时间戳有序，查找效率高
        
        Args:
            cache_key: 缓存键名
            target_timestamp: 目标ROS2时间戳
            max_time_diff: 最大允许时间差（秒），默认2.0秒
            
        Returns:
            (timestamp, data) 或 None，返回时间戳最接近target_timestamp且时间差在max_time_diff内的数据
        """
        if cache_key not in self.data_cache or not self.data_cache[cache_key]:
            if cache_key in self.cache_stats:
                self.cache_stats[cache_key]['misses'] += 1
            return None
        
        cache_list = self.data_cache[cache_key]
        if not cache_list:
            if cache_key in self.cache_stats:
                self.cache_stats[cache_key]['misses'] += 1
            return None
        
        # 使用二分查找找到最接近的数据（缓存已按ROS2时间戳有序）
        # 找到插入位置
        timestamps = [item[0] for item in cache_list]
        pos = bisect.bisect_left(timestamps, target_timestamp)
        
        # 检查左右两个候选
        candidates = []
        
        # 检查左侧（时间戳 <= target）
        if pos > 0:
            left_ts, left_rt, left_data = cache_list[pos - 1]
            left_diff = abs(left_ts - target_timestamp)
            if left_diff <= max_time_diff:
                candidates.append((left_diff, left_ts, left_data))
        
        # 检查当前位置（时间戳 >= target）
        if pos < len(cache_list):
            right_ts, right_rt, right_data = cache_list[pos]
            right_diff = abs(right_ts - target_timestamp)
            if right_diff <= max_time_diff:
                candidates.append((right_diff, right_ts, right_data))
        
        if not candidates:
            if cache_key in self.cache_stats:
                self.cache_stats[cache_key]['misses'] += 1
            return None
        
        # 返回时间差最小的
        best_diff, best_ts, best_data = min(candidates, key=lambda x: x[0])
        if cache_key in self.cache_stats:
            self.cache_stats[cache_key]['hits'] += 1
        return (best_ts, best_data)
    
    def _get_synchronized_data_from_cache(self) -> Dict[str, tuple]:
        """
        自适应时间同步：当所有信号都获取到时，自适应确定cache时间长度，
        找到所有topic时间戳对齐的数据用于推理
        
        Returns:
            Dict[cache_key, (ros2_timestamp, data)]
        """
        required_keys = ['image', 'left_wrist_image', 'right_wrist_image', 'left_hand', 'right_hand', 'head_status', 'waist_status', 'arm_status', 'leg_status']
        
        # 首先检查所有topic是否都有数据
        latest_timestamps = {}
        latest_data = {}
        for cache_key in required_keys:
            data = self._get_latest_from_cache(cache_key)
            if not data:
                logger.warning(f"[时间同步] {cache_key} 无数据，无法进行同步")
                return {}
            ros2_ts, data_obj = data
            latest_timestamps[cache_key] = ros2_ts
            latest_data[cache_key] = data
        
        # 找到所有topic时间戳的最早和最晚值
        all_timestamps = list(latest_timestamps.values())
        earliest_ts = min(all_timestamps)
        latest_ts = max(all_timestamps)
        time_range = latest_ts - earliest_ts
        
        logger.info(f"[时间同步] 所有topic时间戳范围: 最早={earliest_ts:.6f}, 最晚={latest_ts:.6f}, 时间范围={time_range:.3f}s")
        
        # 自适应确定同步基准时间戳
        # 如果时间范围在2秒内，使用最晚的时间戳（保证所有数据都是最新的）
        # 如果时间范围超过2秒，使用中间点（平衡新旧数据）
        max_tolerance = 2.0
        if time_range <= max_tolerance:
            # 时间范围在容差内，使用最晚的时间戳作为基准
            sync_base_ts = latest_ts
            logger.info(f"[时间同步] 时间范围在容差内，使用最晚时间戳作为基准: {sync_base_ts:.6f}")
        else:
            # 时间范围超过容差，使用中间点
            sync_base_ts = (earliest_ts + latest_ts) / 2.0
            logger.info(f"[时间同步] 时间范围超过容差，使用中间点作为基准: {sync_base_ts:.6f}")
        
        # 为每个topic找到最接近基准时间戳的数据
        synchronized_data = {}
        sync_tolerance = max(time_range, max_tolerance)  # 自适应容差：至少覆盖时间范围
        
        for cache_key in required_keys:
            # 在缓存中查找更接近基准时间戳的数据
            data = self._get_data_by_timestamp(cache_key, sync_base_ts, sync_tolerance)
            if data:
                ros2_ts, _ = data
                synchronized_data[cache_key] = data
            else:
                # 如果找不到，使用最新数据（降级策略）
                logger.warning(f"[时间同步] {cache_key}: 未找到时间差在{sync_tolerance:.2f}s内的数据，使用最新数据")
                synchronized_data[cache_key] = latest_data[cache_key]
                ros2_ts = latest_timestamps[cache_key]
            
            # 记录时间差
            time_diff = ros2_ts - sync_base_ts
            if abs(time_diff) < 1.0:
                diff_str = f"{time_diff*1000:.1f}ms"
            else:
                diff_str = f"{time_diff:.3f}s"
            logger.info(f"[时间同步] {cache_key}: 使用ROS2时间戳={ros2_ts:.6f}, 与基准时间差={diff_str}")
        
        return synchronized_data
    
    def _log_cache_time_range(self):
        """
        输出缓存中第一帧和最后一帧各个topic的时间差
        """
        current_time = time.time()
        logger.info("[缓存时间范围] 各topic缓存数据的时间范围:")
        
        for cache_key in self.data_cache:
            cache_list = self.data_cache[cache_key]
            if not cache_list:
                logger.info(f"  {cache_key}: 无数据")
                continue
            
            # 缓存格式：(ros2_timestamp, receive_time, data)
            first_ros2_ts, first_receive_ts, _ = cache_list[0]
            last_ros2_ts, last_receive_ts, _ = cache_list[-1]
            
            # ROS2时间戳的时间范围
            ros2_time_range = last_ros2_ts - first_ros2_ts
            # 接收时间的时间范围
            receive_time_range = last_receive_ts - first_receive_ts
            # 第一帧和最后一帧的接收时间距今多久
            first_receive_age = current_time - first_receive_ts
            last_receive_age = current_time - last_receive_ts
            
            # 格式化时间差：如果小于1秒，显示毫秒；否则显示秒
            if first_receive_age < 1.0:
                first_age_str = f"{first_receive_age*1000:.1f}ms"
            else:
                first_age_str = f"{first_receive_age:.2f}s"
            
            if last_receive_age < 1.0:
                last_age_str = f"{last_receive_age*1000:.1f}ms"
            else:
                last_age_str = f"{last_receive_age:.2f}s"
            
            # 获取缓存统计信息
            stats = self.cache_stats.get(cache_key, {})
            hits = stats.get('hits', 0)
            misses = stats.get('misses', 0)
            evictions = stats.get('evictions', 0)
            hit_rate = (hits / (hits + misses) * 100) if (hits + misses) > 0 else 0.0
            
            logger.info(
                f"  {cache_key}: 第一帧ROS2时间戳={first_ros2_ts:.6f} (接收于{first_age_str}前), "
                f"最后一帧ROS2时间戳={last_ros2_ts:.6f} (接收于{last_age_str}前), "
                f"ROS2时间范围={ros2_time_range:.3f}s, 接收时间范围={receive_time_range:.3f}s, "
                f"数据量={len(cache_list)}/{self.cache_max_size}, "
                f"命中率={hit_rate:.1f}% (hits={hits}, misses={misses}, evictions={evictions})"
            )

    def image_callback(self, msg: Image):
        """图像回调：缓存最新图像（BGR numpy 数组）和时间戳，然后触发推理检查"""
        try:
            # 优先尝试 bgr8，如果失败再根据编码判断
            start_time = time.time()
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception:
                if msg.encoding == "rgb8":
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                else:
                    logger.warning(f"不支持的图像编码: {msg.encoding}")
                    return  # 如果编码不支持，直接返回，不测量时间

            end_time = time.time()
            image_processing_time = (end_time - start_time) * 1000  # 转换为毫秒
            #logger.info(f"[图像处理] 转换时间: {image_processing_time:.2f}ms")
            
            # 保存图像（如果启用，只保存第一帧）
            if self.save_images and not self.image_saved:
                try:
                    timestamp = time.strftime("%Y%m%d_%H%M%S_%f", time.localtime())
                    image_filename = os.path.join(self.image_save_dir, f"image_{timestamp}.jpg")
                    cv2.imwrite(image_filename, cv_image)
                    self.image_saved = True  # 标记已保存，后续不再保存
                    logger.info(f"图像已保存（第一帧）: {image_filename}")
                except Exception as e:
                    logger.warning(f"保存图像失败: {e}")
            
            ######################
            # 1. 图像处理 3s  1s  
            # 2. infer 1s    历史status信息


            # 缓存图像和时间戳（使用ROS2 topic的时间戳）
            image_timestamp = self._extract_ros2_timestamp(msg)
            self.latest_image = cv_image
            self.latest_image_timestamp = image_timestamp
            # 存入缓存
            self._cache_data('image', image_timestamp, cv_image)
            
            # 输出stamp日志用于调试（包含系统时间对比）
            system_time = time.time()
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp = msg.header.stamp
                time_diff = system_time - self.latest_image_timestamp
                #logger.info(
                #    f"[图像] /camera/color/image_raw stamp: sec={stamp.sec}, nanosec={stamp.nanosec}, "
                #    f"ROS2_timestamp={self.latest_image_timestamp:.6f}, system_time={system_time:.6f}, "
                #    f"TTTTTTTTT diff={time_diff*1000:.2f}ms"
                #)
            else:
                logger.warning(
                    f"[图像] /camera/color/image_raw 没有header.stamp，使用系统时间: "
                    f"timestamp={self.latest_image_timestamp:.6f}, system_time={system_time:.6f}"
                )
            # 图像到达后，触发推理检查
            start_time = time.time()
            #self._trigger_inference_if_ready()
            end_time = time.time()
            trigger_time = (end_time - start_time) * 1000  # 转换为毫秒
            #logger.info(f"[触发推理] 检查数据并触发推理时间: {trigger_time:.2f}ms")
        except Exception as e:
            logger.warning(f"图像回调处理失败: {e}")

    def left_wrist_image_callback(self, msg: Image):
        """左腕部相机图像回调：缓存最新图像（BGR numpy 数组）和时间戳"""
        # 在函数最开始就记录日志，确保能看到回调是否被调用
        #logger.info(f"[左腕部相机] 🔔 回调函数被调用！msg type: {type(msg)}, encoding: {getattr(msg, 'encoding', 'N/A')}")
        try:
            #logger.info(f"[左腕部相机] ✅ 回调被触发，收到图像消息，encoding={msg.encoding}, width={msg.width}, height={msg.height}")
            # 优先尝试 bgr8，如果失败再根据编码判断
            start_time = time.time()
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                logger.debug(f"[左腕部相机] 成功使用bgr8转换图像")
            except Exception as e:
                logger.debug(f"[左腕部相机] bgr8转换失败: {e}，尝试rgb8")
                if msg.encoding == "rgb8":
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                    #logger.debug(f"[左腕部相机] 成功使用rgb8转换图像")
                else:
                    #logger.warning(f"左腕部相机不支持的图像编码: {msg.encoding}")
                    return
            
            end_time = time.time()
            image_processing_time = (end_time - start_time) * 1000  # 转换为毫秒
            #logger.info(f"[左腕部相机图像处理] 转换时间: {image_processing_time:.2f}ms")

            # 保存图像（如果启用，只保存第一帧）
            if self.save_images and not self.left_wrist_image_saved:
                try:
                    timestamp = time.strftime("%Y%m%d_%H%M%S_%f", time.localtime())
                    image_filename = os.path.join(self.image_save_dir, f"left_wrist_image_{timestamp}.jpg")
                    cv2.imwrite(image_filename, cv_image)
                    self.left_wrist_image_saved = True  # 标记已保存，后续不再保存
                    logger.info(f"左腕部相机图像已保存（第一帧）: {image_filename}")
                except Exception as e:
                    logger.warning(f"保存左腕部相机图像失败: {e}")

            # 缓存图像和时间戳（使用ROS2 topic的时间戳）
            #logger.debug(f"[左腕部相机] 准备缓存图像，cv_image shape: {cv_image.shape if cv_image is not None else 'None'}")
            left_wrist_timestamp = self._extract_ros2_timestamp(msg)
            self.latest_left_wrist_image = cv_image
            self.latest_left_wrist_image_timestamp = left_wrist_timestamp
            # 存入缓存
            self._cache_data('left_wrist_image', left_wrist_timestamp, cv_image)
            #logger.info(f"[左腕部相机] ✅ 图像已缓存，timestamp={self.latest_left_wrist_image_timestamp:.6f}")
            
            # 输出stamp日志用于调试（包含系统时间对比）
            system_time = time.time()
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp = msg.header.stamp
                time_diff = system_time - self.latest_left_wrist_image_timestamp
                #logger.info(
                #    f"[左腕部相机] /D405/left/color/image_raw stamp: sec={stamp.sec}, nanosec={stamp.nanosec}, "
                #    f"ROS2_timestamp={self.latest_left_wrist_image_timestamp:.6f}, system_time={system_time:.6f}, "
                #    f"diff={time_diff*1000:.2f}ms"
                #)
            else:
                logger.warning(
                    f"[左腕部相机] /D405/left/color/image_raw 没有header.stamp，使用系统时间: "
                    f"timestamp={self.latest_left_wrist_image_timestamp:.6f}, system_time={system_time:.6f}"
                )
            
            # 不再在回调中触发推理，改为定时器触发
            # self._trigger_inference_if_ready()
        except Exception as e:
            logger.error(f"❌ 左腕部相机图像回调处理失败: {e}", exc_info=True)  # 使用exc_info=True显示完整堆栈

    def right_wrist_image_callback(self, msg: Image):
        """右腕部相机图像回调：缓存最新图像（BGR numpy 数组）和时间戳"""
        try:
            #logger.info(f"[右腕部相机] ✅ 回调被触发，收到图像消息，encoding={msg.encoding}, width={msg.width}, height={msg.height}")
            # 优先尝试 bgr8，如果失败再根据编码判断
            start_time = time.time()
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception:
                if msg.encoding == "rgb8":
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                else:
                    logger.warning(f"右腕部相机不支持的图像编码: {msg.encoding}")
                    return
            
            end_time = time.time()
            image_processing_time = (end_time - start_time) * 1000  # 转换为毫秒
            #logger.info(f"[右腕部相机图像处理] 转换时间: {image_processing_time:.2f}ms")

            # 保存图像（如果启用，只保存第一帧）
            if self.save_images and not self.right_wrist_image_saved:
                try:
                    timestamp = time.strftime("%Y%m%d_%H%M%S_%f", time.localtime())
                    image_filename = os.path.join(self.image_save_dir, f"right_wrist_image_{timestamp}.jpg")
                    cv2.imwrite(image_filename, cv_image)
                    self.right_wrist_image_saved = True  # 标记已保存，后续不再保存
                    logger.info(f"右腕部相机图像已保存（第一帧）: {image_filename}")
                except Exception as e:
                    logger.warning(f"保存右腕部相机图像失败: {e}")

            # 缓存图像和时间戳（使用ROS2 topic的时间戳）
            right_wrist_timestamp = self._extract_ros2_timestamp(msg)
            self.latest_right_wrist_image = cv_image
            self.latest_right_wrist_image_timestamp = right_wrist_timestamp
            # 存入缓存
            self._cache_data('right_wrist_image', right_wrist_timestamp, cv_image)
            
            # 输出stamp日志用于调试（包含系统时间对比）
            system_time = time.time()
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp = msg.header.stamp
                time_diff = system_time - self.latest_right_wrist_image_timestamp
                #logger.info(
                #    f"[右腕部相机] /D405/right/color/image_raw stamp: sec={stamp.sec}, nanosec={stamp.nanosec}, "
                #    f"ROS2_timestamp={self.latest_right_wrist_image_timestamp:.6f}, system_time={system_time:.6f}, "
                #    f"diff={time_diff*1000:.2f}ms"
                #)
            else:
                logger.warning(
                    f"[右腕部相机] /D405/right/color/image_raw 没有header.stamp，使用系统时间: "
                    f"timestamp={self.latest_right_wrist_image_timestamp:.6f}, system_time={system_time:.6f}"
                )
            
            # 不再在回调中触发推理，改为定时器触发
            # self._trigger_inference_if_ready()
        except Exception as e:
            logger.warning(f"右腕部相机图像回调处理失败: {e}")

    def left_hand_callback(self, msg: JointState):
        """左手状态回调：缓存数据和时间戳（使用ROS2 topic的时间戳）"""
        hand_timestamp = self._extract_ros2_timestamp(msg)
        self.latest_left_hand = msg
        self.latest_left_hand_timestamp = hand_timestamp
        # 存入缓存
        self._cache_data('left_hand', hand_timestamp, msg)
        
        # 输出stamp日志用于调试（包含系统时间对比）
        system_time = time.time()
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            stamp = msg.header.stamp
            time_diff = system_time - self.latest_left_hand_timestamp
            #logger.info(
            #    f"[左手] /inspire_hand/state/left_hand stamp: sec={stamp.sec}, nanosec={stamp.nanosec}, "
            #    f"ROS2_timestamp={self.latest_left_hand_timestamp:.6f}, system_time={system_time:.6f}, "
            #    f"diff={time_diff*1000:.2f}ms"
            #)
        else:
            logger.warning(
                f"[左手] /inspire_hand/state/left_hand 没有header.stamp，使用系统时间: "
                f"timestamp={self.latest_left_hand_timestamp:.6f}, system_time={system_time:.6f}"
            )

    def right_hand_callback(self, msg: JointState):
        """右手状态回调：缓存数据和时间戳（使用ROS2 topic的时间戳）"""
        hand_timestamp = self._extract_ros2_timestamp(msg)
        self.latest_right_hand = msg
        self.latest_right_hand_timestamp = hand_timestamp
        # 存入缓存
        self._cache_data('right_hand', hand_timestamp, msg)
        
        # 输出stamp日志用于调试（包含系统时间对比）
        system_time = time.time()
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            stamp = msg.header.stamp
            time_diff = system_time - self.latest_right_hand_timestamp
            #logger.info(
            #    f"[右手] /inspire_hand/state/right_hand stamp: sec={stamp.sec}, nanosec={stamp.nanosec}, "
            #    f"ROS2_timestamp={self.latest_right_hand_timestamp:.6f}, system_time={system_time:.6f}, "
            #    f"diff={time_diff*1000:.2f}ms"
            #)
        else:
            logger.warning(
                f"[右手] /inspire_hand/state/right_hand 没有header.stamp，使用系统时间: "
                f"timestamp={self.latest_right_hand_timestamp:.6f}, system_time={system_time:.6f}"
            )

    # -------- 电机状态回调（与 robot_client_ros2.py 中抽取的信息对齐） -------- #
    def _make_motor_status_callback(self, part: str):
        """
        生成电机状态回调函数，缓存数据和时间戳。

        Args:
            part: 'head' | 'waist' | 'arm' | 'leg'
        """

        def _callback(msg):
            # 使用ROS2 topic的时间戳
            timestamp = self._extract_ros2_timestamp(msg)
            
            # 输出stamp日志用于调试（包含系统时间对比）
            topic_name = f"/{part}/status"
            system_time = time.time()
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp = msg.header.stamp
                time_diff = system_time - timestamp
                #logger.info(
                #    f"[{part}] {topic_name} stamp: sec={stamp.sec}, nanosec={stamp.nanosec}, "
                #    f"ROS2_timestamp={timestamp:.6f}, system_time={system_time:.6f}, "
                #    f"diff={time_diff*1000:.2f}ms"
                #)
            else:
                logger.warning(
                    f"[{part}] {topic_name} 没有header.stamp，使用系统时间: "
                    f"timestamp={timestamp:.6f}, system_time={system_time:.6f}"
                )
            
            if part == "head":
                self.latest_head_status = msg
                self.latest_head_status_timestamp = timestamp
                self._cache_data('head_status', timestamp, msg)
            elif part == "waist":
                self.latest_waist_status = msg
                self.latest_waist_status_timestamp = timestamp
                self._cache_data('waist_status', timestamp, msg)
            elif part == "arm":
                self.latest_arm_status = msg
                self.latest_arm_status_timestamp = timestamp
                self._cache_data('arm_status', timestamp, msg)
            elif part == "leg":
                self.latest_leg_status = msg
                self.latest_leg_status_timestamp = timestamp
                self._cache_data('leg_status', timestamp, msg)

        return _callback

    # --------------------------------------------------------------------- #
    # OpenPI 策略客户端初始化与推理
    # --------------------------------------------------------------------- #
    def _initialize_policy_client(self):
        """初始化 OpenPI WebSocket 策略客户端"""
        try:
            self.policy_client = websocket_client_policy.WebsocketClientPolicy(
                host=self.policy_host,
                port=self.policy_port,
            )
            metadata = self.policy_client.get_server_metadata()
            logger.info(f"已连接 OpenPI 策略服务器, metadata={metadata}")
        except Exception as e:
            logger.error(f"连接 OpenPI 策略服务器失败: {e}")
            raise

    def build_state_vector(self) -> np.ndarray:
        """
        构造 32 维状态向量

        说明：
        - 按照 demension.json 的定义，32维状态向量按如下顺序和来源填充：
            0   head_pitch            /head/status name: 2
            1   head_yaw              /head/status name: 3
            2   left_shoulder_pitch   /arm/status name: 11
            3   left_shoulder_roll    /arm/status name: 12
            4   left_shoulder_yaw     /arm/status name: 13
            5   left_elbow_pitch      /arm/status name: 14
            6   left_wrist_yaw        /arm/status name: 15
            7   left_wrist_pitch      /arm/status name: 16
            8   left_wrist_roll       /arm/status name: 17
            9   left_little_finger    /inspire_hand/state/left_hand  name: '1'
           10   left_ring_finger      /inspire_hand/state/left_hand  name: '2'
           11   left_middle_finger    /inspire_hand/state/left_hand  name: '3'
           12   left_fore_finger      /inspire_hand/state/left_hand  name: '4'
           13   left_thumb_bend       /inspire_hand/state/left_hand  name: '5'
           14   left_thumb_rotation   /inspire_hand/state/left_hand  name: '6'
           15   right_shoulder_pitch  /arm/status name: 21
           16   right_shoulder_roll   /arm/status name: 22
           17   right_shoulder_yaw    /arm/status name: 23
           18   right_elbow_pitch     /arm/status name: 24
           19   right_wrist_yaw       /arm/status name: 25
           20   right_wrist_pitch     /arm/status name: 26
           21   right_wrist_roll      /arm/status name: 27
           22   right_little_finger   /inspire_hand/state/right_hand name: '1'
           23   right_ring_finger     /inspire_hand/state/right_hand name: '2'
           24   right_middle_finger   /inspire_hand/state/right_hand name: '3'
           25   right_fore_finger     /inspire_hand/state/right_hand name: '4'
           26   right_thumb_bend      /inspire_hand/state/right_hand name: '5'
           27   right_thumb_rotation  /inspire_hand/state/right_hand name: '6'
           28   waist_yaw             /waist/status name: 31
           29   waist_pitch           /waist/status name: 32
           30   hip_pitch             /leg/status name: 51
           31   knee_pitch            /leg/status name: 52
        """
        state = np.zeros(self.state_dim, dtype=np.float32)

        # ----------------- 头部 2 维：name 2-3 ----------------- #
        if self.latest_head_status is not None and hasattr(self.latest_head_status, "status"):
            try:
                name_to_pos: Dict[int, float] = {}
                for item in self.latest_head_status.status:
                    try:
                        jid = int(item.name)
                        name_to_pos[jid] = float(item.pos)
                    except Exception:
                        continue
                
                # 头部 2 维：2, 3
                if 2 in name_to_pos:
                    state[0] = name_to_pos[2]  # head_pitch
                if 3 in name_to_pos:
                    state[1] = name_to_pos[3]  # head_yaw
            except Exception as e:
                logger.warning(f"从 /head/status 解析关节位置失败，将保持头部 2 维为 0: {e}")

        # ----------------- 利用 /arm/status 填充左右臂 14 维（索引2-15） ----------------- #
        if self.latest_arm_status is not None and hasattr(self.latest_arm_status, "status"):
            try:
                # 将 status 转为 name->pos 的字典，name 为 int
                name_to_pos: Dict[int, float] = {}
                for item in self.latest_arm_status.status:
                    try:
                        jid = int(item.name)
                        name_to_pos[jid] = float(item.pos)
                    except Exception:
                        continue

                # 左臂 7 维：11~17 -> 索引 2~8
                for offset, jid in enumerate(range(11, 18)):
                    if jid in name_to_pos:
                        state[2 + offset] = name_to_pos[jid]

                # 右臂 7 维：21~27 -> 索引 15~21
                for offset, jid in enumerate(range(21, 28)):
                    if jid in name_to_pos:
                        state[15 + offset] = name_to_pos[jid]
            except Exception as e:
                logger.warning(f"从 /arm/status 解析关节位置失败，将保持臂部 14 维为 0: {e}")

        # ----------------- 左手 6 维：name '1'~'6' 的 position（索引9~14） ----------------- #
        if self.latest_left_hand is not None:
            try:
                # name 是字符串数组，如 ['1','2',...]
                name_to_idx: Dict[str, int] = {str(n): i for i, n in enumerate(self.latest_left_hand.name)}
                for k in range(1, 7):
                    key = str(k)
                    if key in name_to_idx:
                        idx = name_to_idx[key]
                        if idx < len(self.latest_left_hand.position):
                            state[8 + k] = float(self.latest_left_hand.position[idx])  # 索引9~14
            except Exception as e:
                logger.warning(f"从 /inspire_hand/state/left_hand 解析手指位置失败，将保持左手 6 维为 0: {e}")

        # ----------------- 右手 6 维：name '1'~'6' 的 position（索引22~27） ----------------- #
        if self.latest_right_hand is not None:
            try:
                name_to_idx: Dict[str, int] = {str(n): i for i, n in enumerate(self.latest_right_hand.name)}
                for k in range(1, 7):
                    key = str(k)
                    if key in name_to_idx:
                        idx = name_to_idx[key]
                        if idx < len(self.latest_right_hand.position):
                            state[21 + k] = float(self.latest_right_hand.position[idx])  # 索引22~27
            except Exception as e:
                logger.warning(f"从 /inspire_hand/state/right_hand 解析手指位置失败，将保持右手 6 维为 0: {e}")

        # ----------------- 腰部 2 维：name 31-32（索引28~29） ----------------- #
        if self.latest_waist_status is not None and hasattr(self.latest_waist_status, "status"):
            try:
                name_to_pos: Dict[int, float] = {}
                for item in self.latest_waist_status.status:
                    try:
                        jid = int(item.name)
                        name_to_pos[jid] = float(item.pos)
                    except Exception:
                        continue
                
                if 31 in name_to_pos:
                    state[28] = name_to_pos[31]  # waist_yaw
                if 32 in name_to_pos:
                    state[29] = name_to_pos[32]  # waist_pitch
            except Exception as e:
                logger.warning(f"从 /waist/status 解析关节位置失败，将保持腰部 2 维为 0: {e}")

        # ----------------- 腿部 2 维：name 51-52（索引30~31） ----------------- #
        if self.latest_leg_status is not None and hasattr(self.latest_leg_status, "status"):
            try:
                name_to_pos: Dict[int, float] = {}
                for item in self.latest_leg_status.status:
                    try:
                        jid = int(item.name)
                        name_to_pos[jid] = float(item.pos)
                    except Exception:
                        continue
                
                if 51 in name_to_pos:
                    state[30] = name_to_pos[51]  # hip_pitch
                if 52 in name_to_pos:
                    state[31] = name_to_pos[52]  # knee_pitch
            except Exception as e:
                logger.warning(f"从 /leg/status 解析关节位置失败，将保持腿部 2 维为 0: {e}")

        return state

    def build_observation(self) -> Optional[Dict[str, Any]]:
        """
        构造 OpenPI 所需的观测字典（事件触发模式）。
        使用时间上最近的status数据（已在_trigger_inference_if_ready中验证时间同步）。
        """
        # 检查所有图像是否都准备好
        if self.latest_image is None:
            return None
        if self.latest_left_wrist_image is None:
            return None
        if self.latest_right_wrist_image is None:
            return None

        # 构建状态向量（使用缓存的最新数据，这些数据的时间戳已在_trigger_inference_if_ready中验证）
        state = self.build_state_vector()
        
        # 记录时间同步信息（DEBUG级别）
        if logger.isEnabledFor(logging.DEBUG) and self.latest_image_timestamp is not None:
            image_time = self.latest_image_timestamp
            time_diffs = []
            if self.latest_arm_status_timestamp is not None:
                time_diffs.append(f"arm: {abs(self.latest_arm_status_timestamp - image_time)*1000:.1f}ms")
            if self.latest_head_status_timestamp is not None:
                time_diffs.append(f"head: {abs(self.latest_head_status_timestamp - image_time)*1000:.1f}ms")
            logger.debug(f"构建观测，时间同步: {', '.join(time_diffs)}")

        # 预处理图像：调整为 224x224, uint8
        # 使用缓存避免重复处理相同图像（降低CPU占用）
        
        # 头部相机图像预处理
        current_image_id = id(self.latest_image)
        if self.cached_processed_image is None or self.cached_image_id != current_image_id:
            processed_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(self.latest_image, self.image_height, self.image_width)
            )
            self.cached_processed_image = processed_image
            self.cached_image_id = current_image_id
        else:
            processed_image = self.cached_processed_image
        
        # 左腕部相机图像预处理
        current_left_wrist_image_id = id(self.latest_left_wrist_image)
        if self.cached_processed_left_wrist_image is None or self.cached_left_wrist_image_id != current_left_wrist_image_id:
            processed_left_wrist_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(self.latest_left_wrist_image, self.image_height, self.image_width)
            )
            self.cached_processed_left_wrist_image = processed_left_wrist_image
            self.cached_left_wrist_image_id = current_left_wrist_image_id
        else:
            processed_left_wrist_image = self.cached_processed_left_wrist_image
        
        # 右腕部相机图像预处理
        current_right_wrist_image_id = id(self.latest_right_wrist_image)
        if self.cached_processed_right_wrist_image is None or self.cached_right_wrist_image_id != current_right_wrist_image_id:
            processed_right_wrist_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(self.latest_right_wrist_image, self.image_height, self.image_width)
            )
            self.cached_processed_right_wrist_image = processed_right_wrist_image
            self.cached_right_wrist_image_id = current_right_wrist_image_id
        else:
            processed_right_wrist_image = self.cached_processed_right_wrist_image

        # 从调试日志可以确认，服务器端的 mylerobot_policy 在 transform 之后
        # 直接访问 data["images"]["cam_high"] 和 data["state"]。
        # 当前配置下，input_transform 没有把 "observation.*" 的键转换为这些字段，
        # 因此这里直接按策略预期构造顶层 "state" 和 "images"。
        # 添加腕部相机图像到观测字典
        observation = {
            "state": state,
            "images": {
                "cam_high": processed_image,  # 头部相机
                "cam_left_wrist": processed_left_wrist_image,  # 左腕部相机
                "cam_right_wrist": processed_right_wrist_image,  # 右腕部相机
            },
            "prompt": self.prompt,
        }

        # 调试信息：只在 DEBUG 级别打印，避免刷屏
        logger.debug(
            "Built observation keys: %s; state.shape=%s, cam_high.shape=%s, cam_left_wrist.shape=%s, cam_right_wrist.shape=%s",
            list(observation.keys()),
            getattr(state, "shape", None),
            getattr(processed_image, "shape", None),
            getattr(processed_left_wrist_image, "shape", None),
            getattr(processed_right_wrist_image, "shape", None),
        )

        return observation

    def infer_actions(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """调用 OpenPI 策略服务器进行推理，返回 (action_horizon, action_dim) 的动作序列"""
        if self.policy_client is None:
            logger.error("policy_client 未初始化")
            return None

        try:
            # 额外调试输出：确认在发送前关键字段是否存在
            if logger.isEnabledFor(logging.DEBUG):
                has_state = "state" in observation
                images_dict = observation.get("images", {})
                has_cam_high = isinstance(images_dict, dict) and "cam_high" in images_dict
                has_cam_left_wrist = isinstance(images_dict, dict) and "cam_left_wrist" in images_dict
                has_cam_right_wrist = isinstance(images_dict, dict) and "cam_right_wrist" in images_dict
                logger.debug(
                    "Sending to policy server: has_state=%s, has_cam_high=%s, has_cam_left_wrist=%s, has_cam_right_wrist=%s",
                    has_state,
                    has_cam_high,
                    has_cam_left_wrist,
                    has_cam_right_wrist,
                )

            # 记录推理开始时间
            t0 = time.time()
            start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))
            logger.info(f"[推理时间] 开始推理: {start_timestamp} (timestamp: {t0:.6f})")
            
            # 暂时注释掉远程推理过程
            result = self.policy_client.infer(observation)
            # 保存推理原始结果到 JSON 文件
            if self.save_infer_result:
                self._save_infer_result(result)
            # 返回零动作数组，避免后续代码报错
            #result = {"actions": np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)}
            
            # 记录推理结束时间
            t1 = time.time()
            infer_time = t1 - t0
            infer_time_ms = infer_time * 1000  # 转换为毫秒
            end_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))
            logger.info(f"[推理时间] 结束推理: {end_timestamp} (timestamp: {t1:.6f})")
            logger.info(f"[推理时间] 推理耗时: {infer_time:.6f}s ({infer_time_ms:.2f}ms)")

            if "actions" not in result:
                logger.error(f"推理结果不包含 'actions' 字段: {result.keys()}")
                return None

            actions = result["actions"]
            actions = np.asarray(actions, dtype=np.float32)

            # 只保留前 self.action_dim 维
            if actions.ndim == 2:
                actions = actions[:, : self.action_dim]
            elif actions.ndim == 1:
                actions = actions[: self.action_dim][None, :]

            # 降低日志级别为 DEBUG，减少CPU占用（INFO级别在10Hz下会产生大量输出）
            logger.debug(
                f"✅ OpenPI 推理成功: 耗时 {infer_time:.3f}s, "
                f"actions.shape={actions.shape}, "
                f"动作范围=[{actions.min():.3f}, {actions.max():.3f}]"
            )
            logger.debug(f"OpenPI 推理完成, 耗时 {infer_time:.3f}s, actions.shape={actions.shape}")

            if self.print_inference_actions:
                logger.info(
                    "推理生成动作 actions shape=%s (行=horizon步, 列=32维):\n%s",
                    actions.shape,
                    np.array2string(actions, precision=6, suppress_small=False, max_line_width=120),
                )
            
            state = observation.get("state")

            # 保存推理结果到文件
            if self.save_actions:
                self._save_actions(state, actions, infer_time)

            if self.save_action_information:
                self._save_action_information_json(actions, infer_time, t0, t1)
            
            return actions
        except Exception as e:
            logger.error(f"OpenPI 推理失败: {e}")
            return None
    
    def _save_actions(self, state: np.ndarray, actions: np.ndarray, infer_time: float):
        """
        将单次推理的 state 和 action 累积到当前会话缓冲区，进程退出时统一写入单文件。
        
        Args:
            state: 推理输入的 32 维状态向量
            actions: 推理结果数组，形状为 (action_horizon, action_dim)
            infer_time: 推理耗时（秒）
        """
        self._session_state_list.append(state.astype(np.float32))
        self._session_npy_list.append(actions.astype(np.float32))
        self._session_meta_list.append((infer_time, actions.shape[0]))
        self._session_infer_times.append(infer_time)

    def _flush_actions_session(self) -> None:
        """将当前会话累积的 state + action 写入一个 .txt 文件。"""
        if not self._session_npy_list:
            return
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            self.action_save_counter += 1
            base = f"session_{timestamp}_{self.action_save_counter:06d}"
            txt_path = os.path.join(self.actions_save_dir, f"{base}.txt")

            all_actions = np.concatenate(self._session_npy_list, axis=0)

            with open(txt_path, "w") as fout:
                fout.write(f"# 会话 state + action 序列（{len(self._session_npy_list)} 次推理）\n")
                fout.write(f"# 合并 action shape: {all_actions.shape}  (total_steps, {all_actions.shape[1]})\n")
                fout.write(f"# 各次推理 horizon: {[h for _, h in self._session_meta_list]}\n")
                fout.write(f"# 各次推理耗时(s): {[f'{t:.3f}' for t in self._session_infer_times]}\n")
                fout.write("# " + "=" * 80 + "\n")

                for i, (state, action) in enumerate(zip(self._session_state_list, self._session_npy_list)):
                    infer_time, horizon = self._session_meta_list[i]
                    if i > 0:
                        fout.write("\n")
                    fout.write(f"# ========== Inference {i+1}/{len(self._session_npy_list)} "
                               f"(horizon={horizon}, infer_time={infer_time:.3f}s) ==========\n")
                    fout.write("# STATE (32-dim)\n")
                    np.savetxt(fout, state, fmt="%.6f", delimiter="\t")
                    fout.write("\n# ACTION ({} steps)\n".format(horizon))
                    np.savetxt(fout, action, fmt="%.6f", delimiter="\t")

            num_inferences = len(self._session_npy_list)
            self._session_state_list.clear()
            self._session_npy_list.clear()
            self._session_meta_list.clear()
            self._session_infer_times.clear()

            logger.info(
                f"会话 state+action 已保存: {txt_path} ({all_actions.shape[0]} action 行, "
                f"{num_inferences} 次推理)"
            )
        except Exception as e:
            logger.warning(f"会话 state+action 保存失败: {e}")


    def _finalize_actions_session(self) -> None:
        """进程退出时自动调用：将累积的 actions 刷到磁盘。"""
        self._flush_actions_session()

    def _save_action_information_json(
        self, actions: np.ndarray, infer_time: float, t0: float, t1: float
    ) -> None:
        """每次推理写一个 JSON：完整 horizon 动作（与策略服务器返回一致，未经关节限位裁剪）。"""
        try:
            ts_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.action_information_counter += 1
            filename = f"action_info_{ts_name}_{self.action_information_counter:06d}.json"
            filepath = os.path.join(self.action_information_dir, filename)

            payload = {
                "description": (
                    "OpenPI policy actions (full horizon rows x 32 dims). "
                    "Values are clipped only when published in _publish_action_vector."
                ),
                "prompt": self.prompt,
                "shape": list(actions.shape),
                "action_dim": int(self.action_dim),
                "infer_time_sec": float(infer_time),
                "wall_clock_start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                "wall_clock_end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
                "unix_t0": float(t0),
                "unix_t1": float(t1),
                "actions": actions.tolist(),
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            logger.debug("已写入 ACTION_INFORMATION: %s", filepath)
        except Exception as e:
            logger.warning("保存 ACTION_INFORMATION JSON 失败: %s", e)

    def _save_infer_result(self, result: Dict[str, Any]):
        """
        将推理原始结果（字典）保存为 JSON 文件。

        Args:
            result: 推理返回的原始字典
        """
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            self.infer_result_counter += 1
            filename = f"infer_result_{timestamp}_{self.infer_result_counter:06d}.json"
            filepath = os.path.join(self.infer_result_save_dir, filename)

            serializable_result = {}
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    serializable_result[k] = v.tolist()
                elif isinstance(v, (np.floating, np.integer)):
                    serializable_result[k] = float(v)
                else:
                    serializable_result[k] = v

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)

            logger.debug(f"已保存推理原始结果: {filepath}")
        except Exception as e:
            logger.warning(f"保存推理原始结果失败: {e}")

    # --------------------------------------------------------------------- #
    # 定时器触发模式：定时器触发推理检查
    # --------------------------------------------------------------------- #
    def timer_callback(self):
        """
        定时器回调函数：每5秒触发一次，检查所有topic数据是否齐全，如果齐全则触发推理
        """
        logger.info("⏰ 定时器触发：开始检查所有topic数据是否齐全")
        
        # 诊断信息：检查各topic数据状态
        topic_status = []
        topic_status.append(f"头部相机: {'✅' if self.latest_image is not None else '❌'}")
        topic_status.append(f"左腕部相机: {'✅' if self.latest_left_wrist_image is not None else '❌'}")
        topic_status.append(f"右腕部相机: {'✅' if self.latest_right_wrist_image is not None else '❌'}")
        topic_status.append(f"左手状态: {'✅' if self.latest_left_hand is not None else '❌'}")
        topic_status.append(f"右手状态: {'✅' if self.latest_right_hand is not None else '❌'}")
        topic_status.append(f"头部状态: {'✅' if self.latest_head_status is not None else '❌'}")
        topic_status.append(f"腰部状态: {'✅' if self.latest_waist_status is not None else '❌'}")
        topic_status.append(f"臂部状态: {'✅' if self.latest_arm_status is not None else '❌'}")
        topic_status.append(f"腿部状态: {'✅' if self.latest_leg_status is not None else '❌'}")
        logger.info(f"[定时器诊断] 各topic数据状态: {', '.join(topic_status)}")
        
        # 额外诊断：检查左腕部相机的详细状态
        if self.latest_left_wrist_image is None:
            logger.warning(f"[定时器诊断] 左腕部相机数据为None，latest_left_wrist_image={self.latest_left_wrist_image}, latest_left_wrist_image_timestamp={self.latest_left_wrist_image_timestamp}")
        if self.latest_right_wrist_image is None:
            logger.warning(f"[定时器诊断] 右腕部相机数据为None，latest_right_wrist_image={self.latest_right_wrist_image}, latest_right_wrist_image_timestamp={self.latest_right_wrist_image_timestamp}")
        
        self._trigger_inference_if_ready()
    
    def _trigger_inference_if_ready(self):
        """
        检查数据是否齐全，如果齐全则触发推理。
        在定时器回调中被调用（每5秒触发一次）。
        必须所有topic数据都存在才触发推理：
        - 头部相机图像
        - 左腕部相机图像
        - 右腕部相机图像
        - 左手状态
        - 右手状态
        - 头部状态
        - 腰部状态
        - 臂部状态
        - 腿部状态
        """
        # 必须有所有图像（头部相机 + 两个腕部相机）
        if self.latest_image is None or self.latest_image_timestamp is None:
            logger.debug("⏳ 等待头部相机图像就绪")
            return
        if self.latest_left_wrist_image is None or self.latest_left_wrist_image_timestamp is None:
            logger.info("⏳ [定时器触发] 等待左腕部相机图像就绪")
            return
        if self.latest_right_wrist_image is None or self.latest_right_wrist_image_timestamp is None:
            logger.info("⏳ [定时器触发] 等待右腕部相机图像就绪")
            return
        
        # 使用头部相机的时间戳作为基准时间
        image_time = self.latest_image_timestamp
        
        # 检查所有必需数据是否齐全，并验证时间同步
        missing_data = []
        time_sync_issues = []
        
        # 检查腕部相机图像时间同步
        left_wrist_time_diff = abs(self.latest_left_wrist_image_timestamp - image_time)
        if left_wrist_time_diff > self.status_time_tolerance:
            time_sync_issues.append(f"左腕部相机时间差: {left_wrist_time_diff*1000:.1f}ms")
        
        right_wrist_time_diff = abs(self.latest_right_wrist_image_timestamp - image_time)
        if right_wrist_time_diff > self.status_time_tolerance:
            time_sync_issues.append(f"右腕部相机时间差: {right_wrist_time_diff*1000:.1f}ms")
        
        # 检查手部状态
        if self.latest_left_hand is None or self.latest_left_hand_timestamp is None:
            missing_data.append("左手状态")
        else:
            time_diff = abs(self.latest_left_hand_timestamp - image_time)
            if time_diff > self.status_time_tolerance:
                time_sync_issues.append(f"左手状态时间差: {time_diff*1000:.1f}ms")
        
        if self.latest_right_hand is None or self.latest_right_hand_timestamp is None:
            missing_data.append("右手状态")
        else:
            time_diff = abs(self.latest_right_hand_timestamp - image_time)
            if time_diff > self.status_time_tolerance:
                time_sync_issues.append(f"右手状态时间差: {time_diff*1000:.1f}ms")
        
        # 检查电机状态（可选，但建议有）
        if self.latest_arm_status is None or self.latest_arm_status_timestamp is None:
            missing_data.append("臂部状态")
        else:
            time_diff = abs(self.latest_arm_status_timestamp - image_time)
            if time_diff > self.status_time_tolerance:
                time_sync_issues.append(f"臂部状态时间差: {time_diff*1000:.1f}ms")
        
        if self.latest_head_status is None or self.latest_head_status_timestamp is None:
            missing_data.append("头部状态")
        else:
            time_diff = abs(self.latest_head_status_timestamp - image_time)
            if time_diff > self.status_time_tolerance:
                time_sync_issues.append(f"头部状态时间差: {time_diff*1000:.1f}ms")
        
        if self.latest_waist_status is None or self.latest_waist_status_timestamp is None:
            missing_data.append("腰部状态")
        else:
            time_diff = abs(self.latest_waist_status_timestamp - image_time)
            if time_diff > self.status_time_tolerance:
                time_sync_issues.append(f"腰部状态时间差: {time_diff*1000:.1f}ms")
        
        if self.latest_leg_status is None or self.latest_leg_status_timestamp is None:
            missing_data.append("腿部状态")
        else:
            time_diff = abs(self.latest_leg_status_timestamp - image_time)
            if time_diff > self.status_time_tolerance:
                time_sync_issues.append(f"腿部状态时间差: {time_diff*1000:.1f}ms")
        
        # 如果有缺失数据，记录警告
        if missing_data:
            logger.info(f"⏳ [定时器触发] 等待数据就绪，缺失: {', '.join(missing_data)}")
            return
        
        # 如果有时间同步问题，记录DEBUG信息（避免刷屏），但继续执行（使用最近的status）
        if time_sync_issues:
            # 计算最大时间差
            max_time_diff = 0
            for issue in time_sync_issues:
                try:
                    diff_str = issue.split(':')[1].split('ms')[0].strip()
                    diff_val = float(diff_str)
                    max_time_diff = max(max_time_diff, diff_val)
                except (ValueError, IndexError):
                    pass
            
            # 只在时间差超过1秒时才记录WARNING，否则记录DEBUG
            if max_time_diff > 1000:
                logger.warning(f"⚠️ 时间同步警告: {', '.join(time_sync_issues)}，将使用时间上最近的status数据")
            else:
                logger.debug(f"时间同步信息: {', '.join(time_sync_issues)}，将使用时间上最近的status数据")
        
        # 所有数据齐全，记录所有topic的时间戳信息差（用于分析）
        logger.info("=" * 80)
        logger.info("📊 所有数据就绪，记录topic时间戳信息差：")
        logger.info(f"  基准时间（头部相机）: {image_time:.6f}")
        logger.info(f"  头部相机: {self.latest_image_timestamp:.6f}, 时间差: 0.00ms (基准)")
        logger.info(f"  左腕部相机: {self.latest_left_wrist_image_timestamp:.6f}, 时间差: {left_wrist_time_diff*1000:.2f}ms")
        logger.info(f"  右腕部相机: {self.latest_right_wrist_image_timestamp:.6f}, 时间差: {right_wrist_time_diff*1000:.2f}ms")
        if self.latest_left_hand_timestamp is not None:
            left_hand_diff = abs(self.latest_left_hand_timestamp - image_time)
            logger.info(f"  左手状态: {self.latest_left_hand_timestamp:.6f}, 时间差: {left_hand_diff*1000:.2f}ms")
        if self.latest_right_hand_timestamp is not None:
            right_hand_diff = abs(self.latest_right_hand_timestamp - image_time)
            logger.info(f"  右手状态: {self.latest_right_hand_timestamp:.6f}, 时间差: {right_hand_diff*1000:.2f}ms")
        if self.latest_arm_status_timestamp is not None:
            arm_diff = abs(self.latest_arm_status_timestamp - image_time)
            logger.info(f"  臂部状态: {self.latest_arm_status_timestamp:.6f}, 时间差: {arm_diff*1000:.2f}ms")
        if self.latest_head_status_timestamp is not None:
            head_diff = abs(self.latest_head_status_timestamp - image_time)
            logger.info(f"  头部状态: {self.latest_head_status_timestamp:.6f}, 时间差: {head_diff*1000:.2f}ms")
        if self.latest_waist_status_timestamp is not None:
            waist_diff = abs(self.latest_waist_status_timestamp - image_time)
            logger.info(f"  腰部状态: {self.latest_waist_status_timestamp:.6f}, 时间差: {waist_diff*1000:.2f}ms")
        if self.latest_leg_status_timestamp is not None:
            leg_diff = abs(self.latest_leg_status_timestamp - image_time)
            logger.info(f"  腿部状态: {self.latest_leg_status_timestamp:.6f}, 时间差: {leg_diff*1000:.2f}ms")
        logger.info("=" * 80)
        
        # 所有数据齐全，先检查是否需要发布初始值
        if not self.initial_action_published:

            logger.info("📤 首次触发：发布初始值，跳过推理")
            self._publish_initial_action()
            self.initial_action_published = True
            # 发布初始值后，继续正常推理
            #self._trigger_inference()
            # pdb.set_trace()
        else:
            # 正常推理流程：从缓存中获取时间同步的数据
            # 自适应时间同步：当所有信号都获取到时，自适应确定cache时间长度，找到所有topic时间戳对齐的数据
            
            synchronized_data = self._get_synchronized_data_from_cache()
            
            # 检查是否所有必需数据都已同步
            required_keys = ['image', 'left_wrist_image', 'right_wrist_image', 'left_hand', 'right_hand', 'arm_status', 'head_status', 'waist_status', 'leg_status']
            missing_sync = [key for key in required_keys if key not in synchronized_data]
            
            if missing_sync:
                logger.warning(f"⏳ 时间同步失败，缺失数据: {', '.join(missing_sync)}，跳过本次推理")
                # 输出缓存数据的时间范围用于调试
                self._log_cache_time_range()
                return
            
            # 更新latest变量为同步后的数据
            if 'image' in synchronized_data:
                self.latest_image_timestamp, self.latest_image = synchronized_data['image']
            
            if 'left_wrist_image' in synchronized_data:
                self.latest_left_wrist_image_timestamp, self.latest_left_wrist_image = synchronized_data['left_wrist_image']
            
            if 'right_wrist_image' in synchronized_data:
                self.latest_right_wrist_image_timestamp, self.latest_right_wrist_image = synchronized_data['right_wrist_image']
            
            if 'left_hand' in synchronized_data:
                self.latest_left_hand_timestamp, self.latest_left_hand = synchronized_data['left_hand']
            
            if 'right_hand' in synchronized_data:
                self.latest_right_hand_timestamp, self.latest_right_hand = synchronized_data['right_hand']
            
            if 'arm_status' in synchronized_data:
                self.latest_arm_status_timestamp, self.latest_arm_status = synchronized_data['arm_status']
            
            if 'head_status' in synchronized_data:
                self.latest_head_status_timestamp, self.latest_head_status = synchronized_data['head_status']
            
            if 'waist_status' in synchronized_data:
                self.latest_waist_status_timestamp, self.latest_waist_status = synchronized_data['waist_status']
            
            if 'leg_status' in synchronized_data:
                self.latest_leg_status_timestamp, self.latest_leg_status = synchronized_data['leg_status']
            
            # 输出缓存数据的时间范围（第一帧和最后一帧的时间差）
            self._log_cache_time_range()
            
            # 输出同步后的各topic的ROS时间戳
            logger.info("=" * 80)
            logger.info("📊 进入推理时同步后的各topic ROS时间戳：")
            if 'image' in synchronized_data:
                ros2_ts, _ = synchronized_data['image']
                logger.info(f"  头部相机 (image): ROS2时间戳={ros2_ts:.6f}")
            if 'left_wrist_image' in synchronized_data:
                ros2_ts, _ = synchronized_data['left_wrist_image']
                logger.info(f"  左腕部相机 (left_wrist_image): ROS2时间戳={ros2_ts:.6f}")
            if 'right_wrist_image' in synchronized_data:
                ros2_ts, _ = synchronized_data['right_wrist_image']
                logger.info(f"  右腕部相机 (right_wrist_image): ROS2时间戳={ros2_ts:.6f}")
            if 'left_hand' in synchronized_data:
                ros2_ts, _ = synchronized_data['left_hand']
                logger.info(f"  左手状态 (left_hand): ROS2时间戳={ros2_ts:.6f}")
            if 'right_hand' in synchronized_data:
                ros2_ts, _ = synchronized_data['right_hand']
                logger.info(f"  右手状态 (right_hand): ROS2时间戳={ros2_ts:.6f}")
            if 'head_status' in synchronized_data:
                ros2_ts, _ = synchronized_data['head_status']
                logger.info(f"  头部状态 (head_status): ROS2时间戳={ros2_ts:.6f}")
            if 'waist_status' in synchronized_data:
                ros2_ts, _ = synchronized_data['waist_status']
                logger.info(f"  腰部状态 (waist_status): ROS2时间戳={ros2_ts:.6f}")
            if 'arm_status' in synchronized_data:
                ros2_ts, _ = synchronized_data['arm_status']
                logger.info(f"  臂部状态 (arm_status): ROS2时间戳={ros2_ts:.6f}")
            if 'leg_status' in synchronized_data:
                ros2_ts, _ = synchronized_data['leg_status']
                logger.info(f"  腿部状态 (leg_status): ROS2时间戳={ros2_ts:.6f}")
            
            # 计算时间戳范围（用于检查同步质量）
            all_timestamps = []
            for cache_key in ['image', 'left_wrist_image', 'right_wrist_image', 'left_hand', 'right_hand', 'head_status', 'waist_status', 'arm_status', 'leg_status']:
                if cache_key in synchronized_data:
                    ros2_ts, _ = synchronized_data[cache_key]
                    all_timestamps.append(ros2_ts)
            
            if len(all_timestamps) > 1:
                earliest_ts = min(all_timestamps)
                latest_ts = max(all_timestamps)
                time_range = latest_ts - earliest_ts
                logger.info(f"  时间戳范围: 最早={earliest_ts:.6f}, 最晚={latest_ts:.6f}, 时间差={time_range*1000:.2f}ms")
            logger.info("=" * 80)
            
            logger.info("✅ 定时器触发：所有数据齐全且已时间同步，使用对齐后的缓存数据开始推理")
            self._trigger_inference()
    
    def _trigger_inference(self):
        """
        触发推理和发布动作（事件驱动模式）。
        在数据齐全时被调用。
        """
        obs = self.build_observation()
        if obs is None:
            logger.warning("构建观测失败，跳过推理")
            return

        actions = self.infer_actions(obs)
        if actions is None or actions.size == 0:
            logger.warning("⚠️ 推理返回空结果，跳过本次发布")
            return

        # 只使用第 0 步动作作为当前控制命令
        # first_action = actions[-1]
        
        # pdb断点：在推理完成后、发布前暂停，便于调试
        #pdb.set_trace()
        for action in actions:
            last_action = action
            # time.sleep(0.033)
            # pdb断点：在推理完成后、发布前暂停，便于调试
            # pdb.set_trace()
            
            # 发布动作向量
            self._publish_action_vector(last_action)
        # 发布动作向量
        # self._publish_action_vector(first_action)
    def destroy_node(self) -> bool:
        """节点销毁时自动刷出未保存的 actions 会话数据。"""
        if self.save_actions:
            self._flush_actions_session()
        return super().destroy_node()

    def _publish_initial_action(self):
        """
        发布初始值（跳过推理，直接使用预设的初始值）。
        """
        interval = 10.0
        prompt_norm = self.prompt.strip().lower()
        logger.info("📤 发布初始值（32维），prompt='%s'，间隔 %.2f s", self.prompt, interval)
        # 启动初值
        # self._publish_action_vector(self.initial_action_1)
        # time.sleep(interval)

        if prompt_norm == "pick up the box":
            # 抓取初值
            self._publish_action_vector(self.initial_action_2)
            time.sleep(interval)
            self._publish_action_vector(self.initial_action_3)
            time.sleep(interval)
        else:
            logger.info("prompt 不是 'pick up the box'，跳过初始值发布。")
    def _publish_action_vector(self, action_vector: np.ndarray):
        """
        发布动作向量到ROS2 topics（通用方法，用于初始值和推理结果）。
        
        Args:
            action_vector: 32维动作向量
        """
        # 按照 demension.json 的映射发布到不同的 topic（32维）
        # 索引 0-1: 头部2个关节 -> /head/cmd_pos (name: 2-3)
        # 索引 2-8: 左臂7个关节 -> /arm/cmd_pos (name: 11-17)
        # 索引 9-14: 左手6个手指 -> /inspire_hand/ctrl/left_hand (name: '1'-'6')
        # 索引 15-21: 右臂7个关节 -> /arm/cmd_pos (name: 21-27)
        # 索引 22-27: 右手6个手指 -> /inspire_hand/ctrl/right_hand (name: '1'-'6')
        # 索引 28-29: 腰部2个关节 -> /waist/cmd_pos (name: 31-32)
        # 索引 30-31: 腿部2个关节 -> /leg/cmd_pos (name: 51-52)
        
        # 发布 /head/cmd_pos (头部2个关节)
        if self.head_cmd_pub is not None and CmdSetMotorPosition is not None and SetMotorPosition is not None:
            try:
                from std_msgs.msg import Header
                
                head_msg = CmdSetMotorPosition()
                head_msg.header = Header()
                head_msg.header.stamp = self.get_clock().now().to_msg()
                head_msg.header.frame_id = "head"
                
                head_msg.cmds = []
                default_speed = 0.1
                default_current = 5.0
                
                # 头部2个关节 (索引0-1, name 2-3)
                for i, name in enumerate(range(2, 4)):
                    cmd_item = SetMotorPosition()
                    cmd_item.name = int(name)
                    raw_pos = float(action_vector[i])
                    if name in self.joint_limits:
                        min_rad, max_rad = self.joint_limits[name]
                        cmd_item.pos = float(np.clip(raw_pos, min_rad, max_rad))
                    else:
                        cmd_item.pos = raw_pos
                    cmd_item.spd = float(default_speed)
                    cmd_item.cur = float(default_current)
                    head_msg.cmds.append(cmd_item)
                
                self.head_cmd_pub.publish(head_msg)
                logger.debug(f"发布 /head/cmd_pos: {len(head_msg.cmds)} 个关节")
            except Exception as e:
                logger.warning(f"发布 /head/cmd_pos 失败: {e}")
        
        # 发布 /arm/cmd_pos (左臂 + 右臂共14个关节)
        # 使用 CmdSetMotorPosition 消息类型，包含 header 和 cmds 字段
        if self.arm_cmd_pub is not None and CmdSetMotorPosition is not None and SetMotorPosition is not None:
            try:
                from std_msgs.msg import Header
                
                arm_msg = CmdSetMotorPosition()
                arm_msg.header = Header()
                arm_msg.header.stamp = self.get_clock().now().to_msg()
                arm_msg.header.frame_id = "arm"  # 与 /arm/status 的 frame_id 保持一致
                
                # 构造 cmds 列表，每个元素是 SetMotorPosition 类型
                arm_msg.cmds = []
                
                # 默认速度值（OpenPI 只输出位置，速度使用固定默认值）
                default_speed = 0.35
                # 固定电流值
                default_current = 5.0
                cur_limit_list = [13.8, 9.5, 3.16, 3.16, 3.16, 2.4, 2.4]

                
                # 左臂7个关节 (索引2-8, name 11-17)
                for i, name in enumerate(range(11, 18)):
                    cmd_item = SetMotorPosition()
                    cmd_item.name = int(name)  # uint16_t 类型
                    # 应用角度限制（将模型输出限制在关节角度范围内）
                    raw_pos = float(action_vector[2 + i])  # 来自32维输出，索引2-8
                    if name in self.joint_limits:
                        min_rad, max_rad = self.joint_limits[name]
                        cmd_item.pos = float(np.clip(raw_pos, min_rad, max_rad))
                        # 如果被限制，记录警告（仅在超出范围时）
                        if raw_pos < min_rad or raw_pos > max_rad:
                            logger.debug(
                                f"关节 {name} 位置超出范围: {raw_pos:.3f} -> "
                                f"限制为 [{min_rad:.3f}, {max_rad:.3f}] -> {cmd_item.pos:.3f}"
                            )
                    else:
                        cmd_item.pos = raw_pos
                    cmd_item.spd = float(default_speed)  # 固定默认速度 0.5
                    cmd_item.cur = float(cur_limit_list[i])  # 固定电流 5.0
                    arm_msg.cmds.append(cmd_item)
                
                # 右臂7个关节 (索引15-21, name 21-27)
                for i, name in enumerate(range(21, 28)):
                    cmd_item = SetMotorPosition()
                    cmd_item.name = int(name)  # uint16_t 类型
                    # 应用角度限制（将模型输出限制在关节角度范围内）
                    raw_pos = float(action_vector[15 + i])  # 来自32维输出，索引15-21
                    if name in self.joint_limits:
                        min_rad, max_rad = self.joint_limits[name]
                        cmd_item.pos = float(np.clip(raw_pos, min_rad, max_rad))
                        # 如果被限制，记录警告（仅在超出范围时）
                        if raw_pos < min_rad or raw_pos > max_rad:
                            logger.debug(
                                f"关节 {name} 位置超出范围: {raw_pos:.3f} -> "
                                f"限制为 [{min_rad:.3f}, {max_rad:.3f}] -> {cmd_item.pos:.3f}"
                            )
                    else:
                        cmd_item.pos = raw_pos
                    cmd_item.spd = float(default_speed)  # 固定默认速度 0.5
                    cmd_item.cur = float(cur_limit_list[i])  # 固定电流 5.0
                    arm_msg.cmds.append(cmd_item)
                
                # 应用位置限制规则：如果左臂关节11和13-17都在[-0.2, 0.2]内，则关节12的绝对值必须>0.2
                # 构建关节位置字典以便检查
                joint_positions = {cmd.name: cmd.pos for cmd in arm_msg.cmds}
                threshold = 0.2
                
                # 检查左臂限制
                left_joint_11 = joint_positions.get(11)
                left_joint_12 = joint_positions.get(12)
                left_joints_13_17 = [joint_positions.get(jid) for jid in range(13, 18)]
                
                if (left_joint_11 is not None and left_joint_12 is not None and 
                    all(pos is not None for pos in left_joints_13_17)):
                    # 检查关节11和13-17是否都在[-0.2, 0.2]范围内
                    if (abs(left_joint_11) <= threshold and 
                        all(abs(pos) <= threshold for pos in left_joints_13_17)):
                        # 如果关节12的绝对值<=0.2，需要调整
                        if abs(left_joint_12) <= threshold:
                            # 调整关节12，使其绝对值>0.2
                            if left_joint_12 >= 0:
                                new_pos_12 = threshold + 0.01  # 设为0.21
                            else:
                                new_pos_12 = -(threshold + 0.01)  # 设为-0.21
                            
                            # 更新arm_msg中关节12的位置
                            for cmd in arm_msg.cmds:
                                if cmd.name == 12:
                                    cmd.pos = float(new_pos_12)
                                    logger.debug(
                                        f"左臂位置限制：关节11和13-17都在[-{threshold}, {threshold}]内，"
                                        f"调整关节12从 {left_joint_12:.3f} 到 {new_pos_12:.3f}"
                                    )
                                    break
                
                # 检查右臂限制（关节21和23-27都在[-0.2, 0.2]内，则关节22的绝对值必须>0.2）
                right_joint_21 = joint_positions.get(21)
                right_joint_22 = joint_positions.get(22)
                right_joints_23_27 = [joint_positions.get(jid) for jid in range(23, 28)]
                
                if (right_joint_21 is not None and right_joint_22 is not None and 
                    all(pos is not None for pos in right_joints_23_27)):
                    # 检查关节21和23-27是否都在[-0.2, 0.2]范围内
                    if (abs(right_joint_21) <= threshold and 
                        all(abs(pos) <= threshold for pos in right_joints_23_27)):
                        # 如果关节22的绝对值<=0.2，需要调整
                        if abs(right_joint_22) <= threshold:
                            # 调整关节22，使其绝对值>0.2
                            if right_joint_22 >= 0:
                                new_pos_22 = threshold + 0.01  # 设为0.21
                            else:
                                new_pos_22 = -(threshold + 0.01)  # 设为-0.21
                            
                            # 更新arm_msg中关节22的位置
                            for cmd in arm_msg.cmds:
                                if cmd.name == 22:
                                    cmd.pos = float(new_pos_22)
                                    logger.debug(
                                        f"右臂位置限制：关节21和23-27都在[-{threshold}, {threshold}]内，"
                                        f"调整关节22从 {right_joint_22:.3f} 到 {new_pos_22:.3f}"
                                    )
                                    break
                
                self.arm_cmd_pub.publish(arm_msg)
                logger.debug(f"发布 /arm/cmd_pos: {len(arm_msg.cmds)} 个关节")
            except Exception as e:
                logger.warning(f"发布 /arm/cmd_pos 失败: {e}")
        elif self.arm_cmd_pub is None:
            logger.debug("CmdSetMotorPosition 不可用，跳过 /arm/cmd_pos 发布")
        
        # 发布 /inspire_hand/ctrl/left_hand (左手6个手指)
        try:
            from std_msgs.msg import Header
            
            left_hand_msg = JointState()
            left_hand_msg.header = Header()
            # 设置 stamp 为 0（sec=0, nanosec=0）
            left_hand_msg.header.stamp = self.get_clock().now().to_msg()
            left_hand_msg.header.stamp.sec = 0
            left_hand_msg.header.stamp.nanosec = 0
            left_hand_msg.header.frame_id = "left_hand"
            left_hand_msg.name = [str(i) for i in range(1, 7)]  # ['1', '2', '3', '4', '5', '6']
            
            # 应用角度限制（将模型输出限制在手指角度范围内）
            left_hand_msg.position = []
            for i, finger_name in enumerate(left_hand_msg.name):
                raw_pos = float(action_vector[9 + i])  # 索引9-14 (32维输出)
                if finger_name in self.hand_joint_limits:
                    min_rad, max_rad = self.hand_joint_limits[finger_name]
                    clipped_pos = float(np.clip(raw_pos, min_rad, max_rad))
                    left_hand_msg.position.append(clipped_pos)
                    # 如果被限制，记录警告（仅在超出范围时）
                    if raw_pos < min_rad or raw_pos > max_rad:
                        logger.debug(
                            f"左手手指 {finger_name} 位置超出范围: {raw_pos:.3f} -> "
                            f"限制为 [{min_rad:.3f}, {max_rad:.3f}] -> {clipped_pos:.3f}"
                        )
                else:
                    left_hand_msg.position.append(raw_pos)
            
            # 设置 velocity 和 effort（单个值）
            left_hand_msg.velocity = [0.1]
            left_hand_msg.effort = [0.2]
            
            self.left_hand_cmd_pub.publish(left_hand_msg)
            logger.debug(f"发布 /inspire_hand/ctrl/left_hand: {len(left_hand_msg.position)} 个手指")
        except Exception as e:
            logger.warning(f"发布 /inspire_hand/ctrl/left_hand 失败: {e}")
        
        # 发布 /inspire_hand/ctrl/right_hand (右手6个手指)
        try:
            from std_msgs.msg import Header
            
            right_hand_msg = JointState()
            right_hand_msg.header = Header()
            # 设置 stamp 为 0（sec=0, nanosec=0）
            right_hand_msg.header.stamp = self.get_clock().now().to_msg()
            right_hand_msg.header.stamp.sec = 0
            right_hand_msg.header.stamp.nanosec = 0
            right_hand_msg.header.frame_id = "right_hand"
            right_hand_msg.name = [str(i) for i in range(1, 7)]  # ['1', '2', '3', '4', '5', '6']
            
            # 应用角度限制（将模型输出限制在手指角度范围内）
            right_hand_msg.position = []
            for i, finger_name in enumerate(right_hand_msg.name):
                raw_pos = float(action_vector[22 + i])  # 索引22-27 (32维输出)
                if finger_name in self.hand_joint_limits:
                    min_rad, max_rad = self.hand_joint_limits[finger_name]
                    clipped_pos = float(np.clip(raw_pos, min_rad, max_rad))
                    right_hand_msg.position.append(clipped_pos)
                    # 如果被限制，记录警告（仅在超出范围时）
                    if raw_pos < min_rad or raw_pos > max_rad:
                        logger.debug(
                            f"右手手指 {finger_name} 位置超出范围: {raw_pos:.3f} -> "
                            f"限制为 [{min_rad:.3f}, {max_rad:.3f}] -> {clipped_pos:.3f}"
                        )
                else:
                    right_hand_msg.position.append(raw_pos)
            
            # 设置 velocity 和 effort（单个值）
            right_hand_msg.velocity = [0.1]
            right_hand_msg.effort = [0.2]
            
            self.right_hand_cmd_pub.publish(right_hand_msg)
            logger.debug(f"发布 /inspire_hand/ctrl/right_hand: {len(right_hand_msg.position)} 个手指")
        except Exception as e:
            logger.warning(f"发布 /inspire_hand/ctrl/right_hand 失败: {e}")
        
        # 发布 /waist/cmd_pos (腰部2个关节)
        if self.waist_cmd_pub is not None and CmdSetMotorPosition is not None and SetMotorPosition is not None:
            try:
                from std_msgs.msg import Header
                
                waist_msg = CmdSetMotorPosition()
                waist_msg.header = Header()
                waist_msg.header.stamp = self.get_clock().now().to_msg()
                waist_msg.header.frame_id = "waist"
                
                waist_msg.cmds = []
                default_speed = 0.35
                default_current = 5.0
                
                # 腰部2个关节 (索引28-29, name 31-32)
                for i, name in enumerate(range(31, 33)):
                    cmd_item = SetMotorPosition()
                    cmd_item.name = int(name)
                    raw_pos = float(action_vector[28 + i])
                    if name in self.joint_limits:
                        min_rad, max_rad = self.joint_limits[name]
                        cmd_item.pos = float(np.clip(raw_pos, min_rad, max_rad))
                    else:
                        cmd_item.pos = raw_pos
                    cmd_item.spd = float(default_speed)
                    cmd_item.cur = float(default_current)
                    waist_msg.cmds.append(cmd_item)
                
                self.waist_cmd_pub.publish(waist_msg)
                logger.debug(f"发布 /waist/cmd_pos: {len(waist_msg.cmds)} 个关节")
            except Exception as e:
                logger.warning(f"发布 /waist/cmd_pos 失败: {e}")
        
        # 发布 /leg/cmd_pos (腿部2个关节)
        if self.leg_cmd_pub is not None and CmdSetMotorPosition is not None and SetMotorPosition is not None:
            try:
                from std_msgs.msg import Header
                
                leg_msg = CmdSetMotorPosition()
                leg_msg.header = Header()
                leg_msg.header.stamp = self.get_clock().now().to_msg()
                leg_msg.header.frame_id = "leg"
                
                leg_msg.cmds = []
                default_speed = 0.35
                default_current = 5.0
                
                # 腿部2个关节 (索引30-31, name 51-52)
                for i, name in enumerate(range(51, 53)):
                    cmd_item = SetMotorPosition()
                    cmd_item.name = int(name)
                    raw_pos = float(action_vector[30 + i])
                    if name in self.joint_limits:
                        min_rad, max_rad = self.joint_limits[name]
                        cmd_item.pos = float(np.clip(raw_pos, min_rad, max_rad))
                    else:
                        cmd_item.pos = raw_pos
                    cmd_item.spd = float(default_speed)
                    cmd_item.cur = float(default_current)
                    leg_msg.cmds.append(cmd_item)
                
                # self.leg_cmd_pub.publish(leg_msg)
                # logger.debug(f"发布 /leg/cmd_pos: {len(leg_msg.cmds)} 个关节")
            except Exception as e:
                logger.warning(f"发布 /leg/cmd_pos 失败: {e}")
        
        # 降低日志级别为 DEBUG，减少CPU占用（INFO级别在10Hz下会产生大量字符串格式化开销）
        # 如果需要监控，可以改为每N次循环输出一次，或使用DEBUG级别
        logger.debug(
            f"📤 发布动作到 ROS2 topics (32维): "
            f"头部2维={[f'{x:.3f}' for x in action_vector[0:2]]}, "
            f"左臂7维={[f'{x:.3f}' for x in action_vector[2:9]]}, "
            f"左手6维={[f'{x:.3f}' for x in action_vector[9:15]]}, "
            f"右臂7维={[f'{x:.3f}' for x in action_vector[15:22]]}, "
            f"右手6维={[f'{x:.3f}' for x in action_vector[22:28]]}, "
            f"腰部2维={[f'{x:.3f}' for x in action_vector[28:30]]}, "
            f"腿部2维={[f'{x:.3f}' for x in action_vector[30:32]]}"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OpenPI ROS2 客户端（订阅 ROS2 topics + 调用策略服务器 + 发布动作）")
    parser.add_argument("--policy-host", type=str, default="localhost", help="OpenPI 策略服务器地址")
    parser.add_argument("--policy-port", type=int, default=8000, help="OpenPI 策略服务器端口")
    parser.add_argument("--control-frequency", type=float, default=10.0, help="控制频率 (Hz)")
    parser.add_argument("--prompt", type=str, default="pick up the box", help="任务提示词")

    args = parser.parse_args()

    if not ROS2_AVAILABLE:
        logger.error("ROS2 未安装或导入失败，无法运行本客户端")
        return

    rclpy.init()
    node: Optional[OpenPIRos2Client] = None

    try:
        node = OpenPIRos2Client(
            policy_host=args.policy_host,
            policy_port=args.policy_port,
            control_frequency=args.control_frequency,
            prompt=args.prompt,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，中止运行")
    except Exception as e:
        logger.error(f"OpenPI ROS2 客户端运行异常: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
