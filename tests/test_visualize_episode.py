from pathlib import Path
import subprocess

import pytest

from tools import visualize_episode as visualizer

render_episode = visualizer.render_episode
select_spread_episode_ids = visualizer.select_spread_episode_ids


def test_select_spread_episode_ids_includes_endpoints():
    assert select_spread_episode_ids(list(range(10)), 5) == [0, 2, 4, 7, 9]


def test_render_rejects_missing_episode(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="episode 000003 parquet"):
        render_episode(
            tmp_path,
            episode_id=3,
            output=tmp_path / "out.mp4",
            camera_keys=["cam_high.color"],
            state_key="observation.qpos",
            action_key="action",
        )


def test_decode_video_frames_reads_av1_with_system_ffmpeg(tmp_path: Path):
    video = tmp_path / "three_frames.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=25:duration=0.12",
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "8",
            "-crf",
            "35",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )

    frames, fps = visualizer.decode_video_frames(video)

    assert fps == pytest.approx(25.0)
    assert len(frames) == 3
    assert all(frame.shape == (48, 64, 3) for frame in frames)
