"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.

uv run scripts/compute_norm_stats.py \
    --config-name robot_32_wrist \
    --repo-id=/pytenpublic-tos-volc-engine/chenhaiying/data/0331/static_putdown



"""

import dataclasses
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class FilterNaNDataset:
    """Skip samples containing NaN in state or actions (or any other loading error).

    Pre-builds a list of valid indices during __init__ so that __getitem__
    transparently maps filtered indices to original dataset indices.
    """

    def __init__(self, dataset):
        self._dataset = dataset
        self._valid_indices = []
        for i in range(len(dataset)):
            try:
                item = dataset[i]
                state = np.asarray(item.get("state", item.get("observation.state", [])), dtype=np.float32)
                actions = np.asarray(item.get("actions", item.get("action", [])), dtype=np.float32)
                if not (np.any(np.isnan(state)) or np.any(np.isnan(actions))):
                    self._valid_indices.append(i)
            except Exception:
                # Skip samples that fail to load (e.g. None fields, corrupt data).
                pass

    def __getitem__(self, index):
        return self._dataset[self._valid_indices[index]]

    def __len__(self):
        return len(self._valid_indices)


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class FallbackImageTransform(transforms.DataTransformFn):
    """Apply the config's data_transforms.inputs chain with zero-image fallback.

    Norm-stats computation needs 'state' and 'actions' to have the same shape as
    training.  The config's data_transforms (e.g. Robot32EgoInputs) do the correct
    end-effector extraction, but they require the image key to be present in the
    sample.  If a camera key is missing (e.g. the dataset only has 'egocentric'
    but the transform expects it under a different name), we inject a zero image
    so the chain can proceed.
    """

    def __init__(self, data_transforms: transforms.Group, model_config: _model.BaseModelConfig):
        self._data_transforms = data_transforms
        self._model_config = model_config

    def __call__(self, x: dict) -> dict:
        try:
            return transforms.compose(self._data_transforms.inputs)(x)
        except KeyError:
            pass
        result = dict(x)
        if "images" in result:
            sample_img = next(iter(result["images"].values()))
            result["images"] = {k: np.zeros_like(sample_img) for k in result["images"]}
        return transforms.compose(self._data_transforms.inputs)(result)


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = FilterNaNDataset(dataset)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            FallbackImageTransform(data_config.data_transforms, model_config),
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    # num_workers must be 0 to avoid multi-process bytecode cache issues.
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            FallbackImageTransform(data_config.data_transforms, model_config),
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    repo_id: str | None = None,
    output_dir: str | None = None,
    max_frames: int | None = None,
):
    """Compute normalization statistics for a config.

    Args:
        config_name: Name of the training config in `openpi.training.config`.
        repo_id: Optional override for the dataset repo id / path. This is useful to reuse one config to
            compute norm stats for multiple datasets without editing `config.py`.
        output_dir: Optional override for where to write the computed `norm_stats.json`. If not provided,
            defaults to `config.assets_dirs / data_config.repo_id` (note: if `repo_id` is an absolute path,
            this default will write into that absolute path).
        max_frames: Optional cap on number of frames used for statistics.
    """
    config = _config.get_config(config_name)

    data_factory = config.data
    if repo_id is not None:
        try:
            data_factory = dataclasses.replace(data_factory, repo_id=repo_id)
        except TypeError as e:
            raise TypeError(
                f"Config '{config_name}' has a data factory of type {type(config.data)!r} "
                "which does not support overriding `repo_id`."
            ) from e

    data_config = data_factory.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, max_frames
        )
    else:
        # Force num_workers=0 to avoid multi-process bytecode cache issues during stats computation.
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, 0, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    if output_dir is not None:
        out = output_dir
    else:
        out = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {out}")
    normalize.save(out, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
