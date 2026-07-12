#!/usr/bin/env python3
"""
SVS Data Preprocessing Script

This script performs all data preprocessing steps offline and saves the results
to Arrow format for fast loading during training.

Supports two data formats:
1. Folder-based format: Each song folder contains metadata.json + audio files + annotation JSONs
2. JSON file format: Single JSON file containing all annotations with relative audio paths

Supports multiple datasets input with unified output.

Preprocessing steps:
1. Load samples from multiple datasets (folder or json format)
2. Tokenize text with SVS tokens (pitch, note, BPM)
3. Encode audio waveforms with AudioVAE to get latent features
4. Compute masks (text_mask, audio_mask, loss_mask)
5. Save all processed data to Arrow format with dataset_name field

Usage:
    python scripts/preprocess_svs_data.py conf/svs_preprocess.yaml

Config format:
    datasets:
      - name: m4singer
        type: json_file
        json_path: /path/to/m4singer.json
        audio_root: /path/to/m4singer/wavs
        song_name_index: 1  # index in '#'-split item_name for song name
      - name: gtsinger
        type: json_file
        json_path: /path/to/gtsinger.json
        audio_root: /path/to/gtsinger/wavs
        song_name_index: 3  # e.g., Chinese#ZH-Alto-1#Breathy#不再见#... -> index 3
      - name: cloudmusic
        type: folder_based
        dataset_root: /path/to/cloudmusic
        # For folder_based, song_name is the first-level folder name

    # (val_songs / val_samples are no longer used at preprocessing time;
    # train/val splitting is performed at training time)

Song name determination:
    - For folder_based: song_name is the first-level folder name under dataset_root
    - For json_file: song_name is extracted from item_name by splitting on '#'
                     and taking the element at song_name_index

The output will be:
    output_dir/
        data-00000-of-NNNNN.arrow
        dataset_info.json
        state.json
        config.json  # Preprocessing config for reproducibility
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch

from vocalrender.preprocessing import (
    SVSPreprocessor,
    load_config,
    load_all_datasets,
    process_and_save,
    process_and_save_multigpu,
)

# Backward-compatible re-exports so that existing ``from preprocess_svs_data
# import ...`` statements in other scripts keep working until they are updated.
from vocalrender.preprocessing import (  # noqa: F401
    create_lightweight_preprocessor,
    rebuild_svs_prompt,
    build_text_tensor,
    estimate_duration_from_notes,
)
# Also re-export convert_words_to_syllables which was defined here previously
from vocalrender.preprocessing.data_loaders import (  # noqa: F401
    _convert_words_to_syllables as convert_words_to_syllables,
)


def main():
    # Check command-line argument
    if len(sys.argv) != 2:
        print("Usage: python scripts/preprocess_svs_data.py <config_path>")
        print("Example: python scripts/preprocess_svs_data.py conf/svs_preprocess.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    print(f"Loading config from: {config_path}")
    params = load_config(config_path)

    print("\nPreprocessing configuration:")
    for key, value in params.items():
        if key == "datasets":
            print(f"  datasets: {len(value)} dataset(s)")
            for ds in value:
                print(f"    - {ds.get('name', 'unknown')}: {ds.get('type', 'unknown')}")
        else:
            print(f"  {key}: {value}")
    print()

    output_dir = Path(params["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save preprocessing config
    save_config = {
        "datasets": params["datasets"],
        "pretrained_path": params["pretrained_path"],
        "sample_rate": params["sample_rate"],
        "max_samples": params["max_samples"],
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(save_config, f, indent=2)

    # NB: the VAE preprocessor is created lazily below — only the single-GPU
    # path needs one in this (parent) process. In multi-GPU mode each worker
    # builds its own, so creating one here just wastes a GPU's worth of memory
    # on cuda:0 for the whole run.

    # Load samples from all datasets. The threaded folder scan already cuts the
    # NFS-bound load to a few minutes, so manifest caching is OFF by default:
    # for cloudmusic-scale datasets a full-sample JSON manifest balloons to many
    # GB (it duplicates every word/pitch/syllable list) and its serialization
    # blocks GPU start. Opt in only by setting ``manifest_dir`` in the config.
    manifest_dir = params.get("manifest_dir")  # None -> caching disabled
    samples = load_all_datasets(
        datasets_config=params["datasets"],
        max_samples=params["max_samples"],
        manifest_dir=manifest_dir,
        refresh_manifest=bool(params.get("refresh_manifest", False)),
    )

    if not samples:
        print("No samples found!")
        return

    print(f"\nTotal samples: {len(samples)}")

    # Show distribution by dataset
    by_dataset = {}
    for s in samples:
        ds = s.get("dataset_name", "unknown")
        by_dataset[ds] = by_dataset.get(ds, 0) + 1
    print(f"Samples by dataset: {by_dataset}")

    # Determine number of GPUs
    num_gpus = params.get("num_gpus", -1)
    if num_gpus == -1:
        num_gpus = torch.cuda.device_count()
    num_gpus = max(1, min(num_gpus, torch.cuda.device_count()))
    use_multigpu = num_gpus > 1 and params["device"] == "cuda"

    if use_multigpu:
        print(f"\nUsing {num_gpus} GPUs for parallel processing")

    # Process and save all data to a single directory
    # Train/val splitting is deferred to training time
    print("\nProcessing all data...")
    if use_multigpu:
        process_and_save_multigpu(
            samples, params,
            output_dir,
            num_gpus=num_gpus,
        )
    else:
        preprocessor = SVSPreprocessor(
            pretrained_path=params["pretrained_path"],
            sample_rate=params["sample_rate"],
            device=params["device"],
        )
        process_and_save(
            samples,
            preprocessor,
            output_dir,
            num_workers=params["num_workers"],
            shard_size=params["shard_size"],
            vae_batch_size=params["vae_batch_size"],
            vae_max_tokens=params["vae_max_tokens"],
        )

    print("\nPreprocessing complete!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
