#!/usr/bin/env python3
"""Create a whole-episode LeRobot v2.1 subset while preserving episode IDs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq


def resolve_episode_assets(root: Path, episode_id: int) -> dict[str, list[Path]]:
    parquet = sorted(root.glob(f"data/**/episode_{episode_id:06d}.parquet"))
    videos = sorted(root.glob(f"videos/**/episode_{episode_id:06d}.mp4"))
    if len(parquet) != 1:
        raise ValueError(f"episode {episode_id}: expected one parquet, found {len(parquet)}")
    if not videos:
        raise ValueError(f"episode {episode_id}: found no videos")
    return {"parquet": parquet, "videos": videos}


def _filter_jsonl(source: Path, destination: Path, episode_ids: set[int]) -> int:
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    selected = [record for record in records if int(record["episode_index"]) in episode_ids]
    destination.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in selected)
    )
    return len(selected)


def create_subset(src: Path, dst: Path, episode_ids: Sequence[int]) -> dict[str, object]:
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    requested = sorted(set(map(int, episode_ids)))
    if len(requested) != len(episode_ids) or not requested:
        raise ValueError("episode IDs must be non-empty and unique")
    available = {
        int(json.loads(line)["episode_index"])
        for line in (src / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    }
    for episode_id in requested:
        if episode_id not in available:
            raise ValueError(f"episode {episode_id} is not present")

    assets = {episode_id: resolve_episode_assets(src, episode_id) for episode_id in requested}
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{dst.name}.tmp-", dir=dst.parent))
    try:
        shutil.copytree(src / "meta", temporary / "meta")
        for stale in ("stats.json", "relative_stats.json", "split.json", "preparation_manifest.json"):
            (temporary / "meta" / stale).unlink(missing_ok=True)
        count = _filter_jsonl(
            src / "meta/episodes.jsonl",
            temporary / "meta/episodes.jsonl",
            set(requested),
        )
        assert count == len(requested)
        episode_stats = src / "meta/episodes_stats.jsonl"
        if episode_stats.is_file():
            _filter_jsonl(
                episode_stats,
                temporary / "meta/episodes_stats.jsonl",
                set(requested),
            )

        frames = 0
        video_count = 0
        chunks = set()
        for episode_assets in assets.values():
            parquet_source = episode_assets["parquet"][0]
            parquet_destination = temporary / parquet_source.relative_to(src)
            parquet_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parquet_source, parquet_destination)
            frames += pq.read_metadata(parquet_source).num_rows
            for video_source in episode_assets["videos"]:
                video_destination = temporary / video_source.relative_to(src)
                video_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_source, video_destination)
                video_count += 1
                chunks.add(video_source.parts[video_source.parts.index("videos") + 1])

        info_path = temporary / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["total_episodes"] = len(requested)
        info["total_frames"] = frames
        info["total_videos"] = video_count
        info["total_chunks"] = len(chunks)
        info["splits"] = {"train": f"0:{len(requested)}"}
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        report = {
            "source": str(src.resolve()),
            "destination": str(dst.resolve()),
            "episode_ids": requested,
            "episodes": len(requested),
            "frames": frames,
            "videos": video_count,
            "statistics_reset": True,
        }
        (temporary / "meta/subset_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        os.replace(temporary, dst)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--episode-ids", required=True, nargs="+", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(create_subset(args.src, args.dst, args.episode_ids), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
