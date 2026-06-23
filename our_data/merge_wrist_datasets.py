#!/usr/bin/env python3
"""
合并多个 LeRobot 数据集，保留 Arrow metadata（避免 Image → dict 错误）

核心功能：
1. 使用 PyArrow API 直接操作 parquet，保留 extension type metadata
2. 正确更新所有索引：episode_index, frame_index, global index
3. 合并所有元数据文件
4. 支持命令行参数指定数据集路径

使用示例：
    uv run python our_data/merge_wrist_datasets.py --src \
    /iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data/grab_detail_wrist_0123_train_choose2 \
    /iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data/0206_1_grab_train_choose1 \
    --dst /iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data/0123_0206_choose2 \
    --task-name "pick up the box"
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm


def _replace_int64_column(table: pa.Table, col_name: str, values) -> pa.Table:
    """用 int64 数组替换指定列。"""
    col_idx = table.schema.get_field_index(col_name)
    if col_idx < 0:
        raise KeyError(f"Missing required column: {col_name}")
    arr = pa.array(values, type=pa.int64())
    return table.set_column(col_idx, col_name, arr)


def load_dataset_info(dataset_path: Path) -> Tuple[int, int, str]:
    """加载数据集基本信息"""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    episodes = info["total_episodes"]
    frames = info["total_frames"]
    
    # 尝试从 tasks.jsonl 获取任务描述
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    task = "default_task"
    if tasks_path.exists():
        with open(tasks_path, 'r') as f:
            first_line = f.readline()
            if first_line:
                task_data = json.loads(first_line)
                task = task_data.get("task", "default_task")
    
    return episodes, frames, task


def create_output_structure(output_dir: Path):
    """创建输出目录结构"""
    print("\n" + "=" * 80)
    print("创建输出目录结构")
    print("=" * 80)
    
    # 删除已存在的输出目录
    if output_dir.exists():
        print(f"删除已存在的目录: {output_dir}")
        shutil.rmtree(output_dir)
    
    # 创建新目录
    output_dir.mkdir(parents=True)
    (output_dir / "meta").mkdir()
    (output_dir / "data").mkdir()
    (output_dir / "data" / "chunk-000").mkdir()
    
    print(f"✓ 已创建: {output_dir}")


def merge_info_json(src_datasets: List[Path], output_dir: Path, task_name: str):
    """合并 info.json"""
    print("\n" + "=" * 80)
    print("合并 info.json")
    print("=" * 80)
    
    # 读取第一个数据集的 info.json 作为模板
    template_path = src_datasets[0] / "meta" / "info.json"
    with open(template_path, 'r') as f:
        info = json.load(f)
    
    # 统计所有数据集
    total_episodes = 0
    total_frames = 0
    for dataset_path in src_datasets:
        episodes, frames, _ = load_dataset_info(dataset_path)
        total_episodes += episodes
        total_frames += frames
    
    # 更新统计信息
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = 1  # 合并后统一为一个任务
    info["splits"]["train"] = f"0:{total_episodes}"
    
    # 保存
    output_path = output_dir / "meta" / "info.json"
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✓ 已保存: {output_path}")
    print(f"  总 episodes: {total_episodes}")
    print(f"  总 frames: {total_frames}")
    print(f"  总 tasks: 1")


def merge_tasks_jsonl(output_dir: Path, task_name: str):
    """创建 tasks.jsonl - 统一任务"""
    print("\n" + "=" * 80)
    print("创建 tasks.jsonl")
    print("=" * 80)
    
    output_path = output_dir / "meta" / "tasks.jsonl"
    
    # 只有一个统一的任务
    task_entry = {
        "task_index": 0,
        "task": task_name
    }
    
    with open(output_path, 'w') as f:
        f.write(json.dumps(task_entry) + '\n')
    
    print(f"✓ 已保存: {output_path}")
    print(f"  任务: {task_name}")


def merge_episodes_jsonl(src_datasets: List[Path], output_dir: Path, task_name: str):
    """合并 episodes.jsonl"""
    print("\n" + "=" * 80)
    print("合并 episodes.jsonl")
    print("=" * 80)
    
    output_path = output_dir / "meta" / "episodes.jsonl"
    
    global_episode_index = 0
    
    with open(output_path, 'w') as f_out:
        for dataset_path in src_datasets:
            input_path = dataset_path / "meta" / "episodes.jsonl"
            
            print(f"\n处理数据集: {dataset_path.name}")
            
            with open(input_path, 'r') as f_in:
                for line in tqdm(f_in, desc=f"  合并 {dataset_path.name}"):
                    episode = json.loads(line)
                    
                    # 更新索引和任务
                    episode["episode_index"] = global_episode_index
                    episode["tasks"] = [task_name]  # 统一任务名
                    
                    f_out.write(json.dumps(episode) + '\n')
                    global_episode_index += 1
    
    print(f"\n✓ 已保存: {output_path}")
    print(f"  总 episodes: {global_episode_index}")


def merge_episodes_stats_jsonl(src_datasets: List[Path], output_dir: Path):
    """合并 episodes_stats.jsonl"""
    print("\n" + "=" * 80)
    print("合并 episodes_stats.jsonl")
    print("=" * 80)
    
    output_path = output_dir / "meta" / "episodes_stats.jsonl"
    
    global_episode_index = 0
    
    with open(output_path, 'w') as f_out:
        for dataset_path in src_datasets:
            input_path = dataset_path / "meta" / "episodes_stats.jsonl"
            
            if not input_path.exists():
                print(f"  ⚠️  {dataset_path.name}: episodes_stats.jsonl 不存在，跳过")
                continue
            
            print(f"\n处理数据集: {dataset_path.name}")
            
            with open(input_path, 'r') as f_in:
                for line in tqdm(f_in, desc=f"  合并 {dataset_path.name}"):
                    stats = json.loads(line)
                    
                    # 更新 episode_index
                    stats["episode_index"] = global_episode_index
                    
                    f_out.write(json.dumps(stats) + '\n')
                    global_episode_index += 1
    
    print(f"\n✓ 已保存: {output_path}")
    print(f"  总 episodes: {global_episode_index}")


def merge_parquet_files(src_datasets: List[Path], output_dir: Path):
    """
    合并 parquet 文件，使用 PyArrow 保留 Arrow metadata（避免 Image → dict 错误）
    
    关键：
    1. 使用 PyArrow API 而不是 pandas
    2. 正确更新 episode_index, frame_index, global index
    3. 保留 Arrow extension type metadata（特别是 Image）
    """
    print("\n" + "=" * 80)
    print("合并 parquet 文件（保留 Arrow metadata）")
    print("=" * 80)
    
    global_index = 0
    global_episode_index = 0
    
    for dataset_path in src_datasets:
        input_dir = dataset_path / "data" / "chunk-000"
        
        print(f"\n处理数据集: {dataset_path.name}")
        
        # 获取所有 parquet 文件
        parquet_files = sorted(input_dir.glob("episode_*.parquet"))
        
        for parquet_file in tqdm(parquet_files, desc=f"  处理 {dataset_path.name}"):
            # 对嵌套/扩展列使用流式批读取，避免 read_table 在某些 pyarrow 版本下的 chunked nested 转换报错
            pf = pq.ParquetFile(parquet_file)
            schema = pf.schema_arrow

            output_path = output_dir / "data" / "chunk-000" / f"episode_{global_episode_index:06d}.parquet"
            writer = pq.ParquetWriter(output_path, schema=schema)

            try:
                for batch in pf.iter_batches(batch_size=512, use_threads=True):
                    table = pa.Table.from_batches([batch])
                    batch_rows = table.num_rows
                    if batch_rows == 0:
                        continue

                    table = _replace_int64_column(table, "episode_index", [global_episode_index] * batch_rows)
                    table = _replace_int64_column(table, "index", range(global_index, global_index + batch_rows))
                    table = _replace_int64_column(table, "task_index", [0] * batch_rows)

                    writer.write_table(table)
                    global_index += batch_rows
            finally:
                writer.close()

            # frame_index 在每个 episode 内部从 0 开始，不需要修改
            global_episode_index += 1
    
    print(f"\n✓ 已保存所有 parquet 文件")
    print(f"  总 episodes: {global_episode_index}")
    print(f"  总 frames: {global_index}")
    print(f"  Arrow metadata 已保留（图像将正确识别为 PIL.Image）")


def main():
    parser = argparse.ArgumentParser(
        description="合并多个 LeRobot 数据集，保留 Arrow metadata"
    )
    parser.add_argument(
        "--src",
        required=True,
        type=str,
        nargs="+",
        help="源数据集路径列表（空格分隔）"
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=str,
        help="输出数据集路径"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="default_task",
        help="统一的任务名称（默认: default_task）"
    )
    args = parser.parse_args()
    
    # 转换为 Path 对象
    src_datasets = [Path(p) for p in args.src]
    output_dir = Path(args.dst)
    
    # 验证源数据集存在
    print("=" * 80)
    print("合并 LeRobot 数据集")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"任务名称: {args.task_name}")
    print()
    
    total_episodes = 0
    total_frames = 0
    
    for dataset_path in src_datasets:
        if not dataset_path.exists():
            print(f"❌ 错误: 数据集不存在: {dataset_path}")
            return
        
        episodes, frames, task = load_dataset_info(dataset_path)
        print(f"  - {dataset_path.name}: {episodes} episodes, {frames:,} frames - '{task}'")
        total_episodes += episodes
        total_frames += frames
    
    print()
    print(f"合并后总计: {total_episodes} episodes, {total_frames:,} frames")
    print(f"特性: 使用 PyArrow API 保留 Arrow extension type metadata")
    
    # 执行合并
    create_output_structure(output_dir)
    merge_info_json(src_datasets, output_dir, args.task_name)
    merge_tasks_jsonl(output_dir, args.task_name)
    merge_episodes_jsonl(src_datasets, output_dir, args.task_name)
    merge_episodes_stats_jsonl(src_datasets, output_dir)
    merge_parquet_files(src_datasets, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ 合并完成！")
    print("=" * 80)
    print(f"输出数据集: {output_dir}")
    print(f"  总 episodes: {total_episodes}")
    print(f"  总 frames: {total_frames:,}")
    print(f"  任务: {args.task_name}")
    print(f"  Arrow metadata 已保留 ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()
