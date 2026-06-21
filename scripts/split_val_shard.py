"""Split one WebDataset tar into disjoint validation shards by sample key."""

from __future__ import annotations

import argparse
import tarfile
from contextlib import ExitStack
from pathlib import Path


def split_webdataset(source: Path, output_dir: Path, num_shards: int) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Source tar not found: {source}")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / f"val_part_{index}.tar" for index in range(num_shards)
    ]
    for path in output_paths:
        path.unlink(missing_ok=True)

    sample_to_shard: dict[str, int] = {}
    with ExitStack() as stack:
        source_tar = stack.enter_context(tarfile.open(source, "r"))
        output_tars = [
            stack.enter_context(tarfile.open(path, "w")) for path in output_paths
        ]

        for member in source_tar:
            if not member.isfile():
                continue
            sample_key = Path(member.name).name.split(".", 1)[0]
            shard_index = sample_to_shard.setdefault(
                sample_key, len(sample_to_shard) % num_shards
            )
            file_object = source_tar.extractfile(member)
            output_tars[shard_index].addfile(member, file_object)

    print(
        f"Split {len(sample_to_shard)} samples from {source} into "
        f"{num_shards} shards under {output_dir}."
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input WebDataset tar.")
    parser.add_argument("output_dir", type=Path, help="Output directory.")
    parser.add_argument("num_shards", type=int, help="Number of output shards.")
    return parser


if __name__ == "__main__":
    arguments = build_argparser().parse_args()
    split_webdataset(
        arguments.source,
        arguments.output_dir,
        arguments.num_shards,
    )
