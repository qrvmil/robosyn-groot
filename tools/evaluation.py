"""Launch isolated GR00T inference and real RoboSyn closed-loop evaluation."""

from __future__ import annotations

import json
import os
import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pipeline import GROOT_COMMIT, Workspace, evaluation_episodes_for_task


ROBOSYN_SOURCE = Path("/opt/robosyn/RoboSynChallenge")
ROBOSYN_COMMIT = "93f95f898b76548cc259d20e2b90860a6f79120d"
# v0.2.3 predates the public ``embodichain_tasks`` package imported by the
# pinned RoboSyn checkout.  v0.2.4 is the first release that contains it.
EMBODICHAIN_COMMIT = "9ebee30011f378f94a7cbe78b01d8c2eacba231a"


@dataclass(frozen=True)
class EvaluationSpec:
    task: str
    checkpoint: Path
    checkpoint_sha256: str
    dataset_repo: str
    dataset_revision: str
    output_dir: Path
    episodes: int
    port: int

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.episodes))

    @property
    def protocol_source(self) -> str:
        if self.episodes == 1:
            return "rollout_smoke"
        if self.episodes == 50:
            return "fallback_50_seeds_0_49"
        if self.episodes == 20:
            return "user_override_20_seeds_0_19"
        return f"configured_{self.episodes}_seeds_0_{self.episodes - 1}"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def load_evaluation_spec(
    workspace: Workspace, task: str, *, smoke: bool, port: int = 5555
) -> EvaluationSpec:
    training = _read_json(workspace.run_dir(task) / "training_result.json")
    if training.get("status") != "pass" or training.get("reload_status") != "pass":
        raise RuntimeError(f"task {task} has no verified final checkpoint")
    checkpoint = Path(str(training["checkpoint"]))
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"verified checkpoint is missing: {checkpoint}")
    registry = _read_json(workspace.root / "configs/tasks/registry.json")
    entry = registry["tasks"][task]
    return EvaluationSpec(
        task=task,
        checkpoint=checkpoint,
        checkpoint_sha256=str(training["checkpoint_sha256"]),
        dataset_repo=str(entry["hf_repository"]),
        dataset_revision=str(entry["resolved_hf_revision"]),
        output_dir=workspace.root / "eval" / task / ("smoke" if smoke else ""),
        episodes=(
            1
            if smoke
            else int(entry.get("evaluation_episodes", evaluation_episodes_for_task(task)))
        ),
        port=port,
    )


def evaluation_environment(workspace: Workspace, spec: EvaluationSpec) -> dict[str, str]:
    return {
        **os.environ,
        "ROBOSYN_WORK_ROOT": str(workspace.root),
        "ROBOSYN_SIM_PYTHON": str(workspace.root / ".venvs/robosyn/bin/python"),
        "EMBODICHAIN_ROOT": str(workspace.root / "repos/EmbodiChain"),
        "EMBODICHAIN_DATA_ROOT": str(workspace.root / "cache/embodichain_data"),
        "EMBODICHAIN_DATASET_ROOT": str(
            workspace.root / "cache/embodichain_datasets"
        ),
        "ROBOSYN_CHECKPOINT_SHA256": spec.checkpoint_sha256,
        "ROBOSYN_DATASET_REPO": spec.dataset_repo,
        "ROBOSYN_DATASET_REVISION": spec.dataset_revision,
        "ROBOSYN_SIMULATOR_COMMIT": ROBOSYN_COMMIT,
        "ROBOSYN_GROOT_COMMIT": GROOT_COMMIT,
        "ROBOSYN_EVALUATION_COMMAND": f"robosyn-eval {spec.task}",
        "ROBOSYN_PROTOCOL_SOURCE": spec.protocol_source,
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": str(workspace.root / "cache/huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(workspace.root / "cache/huggingface/hub"),
        "TOKENIZERS_PARALLELISM": "false",
    }


def build_server_command(workspace: Workspace, spec: EvaluationSpec) -> list[str]:
    return [
        str(workspace.groot_link / ".venv/bin/python"),
        str(workspace.groot_link / "gr00t/eval/run_gr00t_server.py"),
        "--model-path",
        str(spec.checkpoint),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--device",
        "cuda",
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.port),
    ]


def build_simulator_command(spec: EvaluationSpec) -> list[str]:
    return [
        str(ROBOSYN_SOURCE / "policy/groot/eval.sh"),
        spec.task,
        str(spec.checkpoint),
        str(spec.output_dir),
        str(spec.episodes),
        str(spec.port),
    ]


def validate_rollout_artifacts(output_dir: Path, episodes: int) -> Path:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"evaluation produced no summary: {summary_path}")
    summary = _read_json(summary_path)
    if summary.get("inference_latency_scope") != "model_rpc_only":
        raise RuntimeError(
            f"evaluation inference latency is not model-RPC-only: {summary_path}"
        )
    if int(summary.get("number_of_episodes", 0)) != episodes:
        raise RuntimeError(f"evaluation incomplete: {summary_path}")
    if int(summary.get("requested_episodes", 0)) != episodes:
        raise RuntimeError(f"evaluation requested episode count is invalid: {summary_path}")
    expected_seeds = list(range(episodes))
    if summary.get("seed_list") != expected_seeds:
        raise RuntimeError(
            f"evaluation seed list is not the canonical 0..{episodes - 1}: {summary_path}"
        )
    results_path = output_dir / "episode_results.jsonl"
    if not results_path.is_file():
        raise RuntimeError(f"evaluation produced no episode results: {results_path}")
    results = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    if len(results) != episodes:
        raise RuntimeError(f"evaluation has {len(results)}/{episodes} episode records")
    invalid = [
        result
        for result in results
        if result.get("error") is not None
        or int(result.get("inference_calls", 0)) < 1
        or int(result.get("episode_length", 0)) < 1
    ]
    if invalid:
        raise RuntimeError(
            f"evaluation contains {len(invalid)} non-rollout/error episodes; see {results_path}"
        )
    result_seeds = [result.get("seed") for result in results]
    if result_seeds != expected_seeds:
        raise RuntimeError(
            f"evaluation seed list is not the canonical 0..{episodes - 1}: {results_path}"
        )
    videos = sorted((output_dir / "videos").glob("*.mp4"))
    if len(videos) < episodes:
        raise RuntimeError(f"evaluation produced {len(videos)}/{episodes} rollout videos")
    video_evidence = []
    for video in videos:
        probed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames:format=duration",
                "-of",
                "json",
                str(video),
            ],
            capture_output=True,
            text=True,
        )
        try:
            metadata = json.loads(probed.stdout) if probed.returncode == 0 else {}
            stream = metadata["streams"][0]
            valid = (
                bool(stream.get("codec_name"))
                and int(stream.get("width", 0)) > 0
                and int(stream.get("height", 0)) > 0
                and float(metadata.get("format", {}).get("duration", 0)) > 0
            )
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            valid = False
        if not valid:
            raise RuntimeError(
                f"invalid rollout video: {video}; ffprobe: {probed.stderr.strip()}"
            )
        video_evidence.append(
            {
                "path": str(video),
                "size_bytes": video.stat().st_size,
                "codec": stream["codec_name"],
                "width": int(stream["width"]),
                "height": int(stream["height"]),
                "frames": stream.get("nb_frames"),
                "duration_seconds": float(metadata["format"]["duration"]),
            }
        )
    evidence_path = output_dir / "video_validation.json"
    temporary = evidence_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"status": "pass", "validated_videos": len(video_evidence), "videos": video_evidence},
            indent=2,
        )
        + "\n"
    )
    os.replace(temporary, evidence_path)
    return summary_path


def _wait_for_server(process: subprocess.Popen, port: int, timeout_s: int = 300) -> None:
    sys.path.insert(0, str(ROBOSYN_SOURCE))
    from policy.groot.client import GrootPolicyClient

    deadline = time.monotonic() + timeout_s
    with GrootPolicyClient("127.0.0.1", port, timeout_ms=1000) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"GR00T server exited before ready: {process.returncode}")
            if client.ping():
                return
            time.sleep(2)
    raise TimeoutError(f"GR00T server was not ready within {timeout_s}s")


def _stop_server(process: subprocess.Popen, port: int) -> None:
    if process.poll() is not None:
        return
    sys.path.insert(0, str(ROBOSYN_SOURCE))
    from policy.groot.client import GrootPolicyClient

    try:
        with GrootPolicyClient("127.0.0.1", port, timeout_ms=3000) as client:
            client.call_endpoint("kill", requires_input=False)
        process.wait(timeout=10)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run_with_fresh_server_retries(operation, *, attempts: int = 3):
    """Retry a failed evaluation attempt with a newly constructed server.

    The operation owns one complete server lifecycle. RoboSyn persists only
    successful episode records, so a later attempt resumes at the failed seed.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            return operation(attempt)
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < attempts:
                print(
                    f"Evaluation attempt {attempt}/{attempts} failed; "
                    "restarting a fresh GR00T server and resuming.",
                    flush=True,
                )
    raise RuntimeError(
        f"evaluation failed after {attempts} fresh-server attempts: "
        + " | ".join(errors)
    ) from None


def run_closed_loop_evaluation(workspace: Workspace, task: str, *, smoke: bool) -> Path:
    spec = load_evaluation_spec(workspace, task, smoke=smoke)
    summary_path = spec.output_dir / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if (
            int(summary.get("number_of_episodes", 0)) == spec.episodes
            and summary.get("inference_latency_scope") == "model_rpc_only"
        ):
            validated = validate_rollout_artifacts(spec.output_dir, spec.episodes)
            print(f"Closed-loop evaluation already complete: {validated}")
            return validated

    sim_python = workspace.root / ".venvs/robosyn/bin/python"
    if not sim_python.is_file():
        raise RuntimeError(f"RoboSyn simulator venv is not ready: {sim_python}")
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    logs = spec.output_dir / "rollout_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = evaluation_environment(workspace, spec)
    server_command = build_server_command(workspace, spec)
    simulator_command = build_simulator_command(spec)
    (logs / "server_command.txt").write_text(" ".join(server_command) + "\n")
    (logs / "evaluation_command.txt").write_text(" ".join(simulator_command) + "\n")

    def run_attempt(attempt: int) -> None:
        with (logs / "server.log").open("a") as server_log:
            server_log.write(f"\n=== fresh-server attempt {attempt}/3 ===\n")
            server_log.flush()
            server = subprocess.Popen(
                server_command,
                cwd=workspace.groot_link,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )
            try:
                _wait_for_server(server, spec.port)
                with (logs / "evaluation.log").open("a") as eval_log:
                    eval_log.write(f"\n=== evaluation attempt {attempt}/3 ===\n")
                    eval_log.flush()
                    evaluated = subprocess.run(
                        simulator_command,
                        cwd=ROBOSYN_SOURCE,
                        env=env,
                        stdout=eval_log,
                        stderr=subprocess.STDOUT,
                    )
                if evaluated.returncode != 0:
                    raise RuntimeError(
                        f"RoboSyn closed-loop evaluator exited {evaluated.returncode}; "
                        f"see {logs / 'evaluation.log'}"
                    )
            finally:
                _stop_server(server, spec.port)

    run_with_fresh_server_retries(run_attempt, attempts=3)

    return validate_rollout_artifacts(spec.output_dir, spec.episodes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    workspace = Workspace(Path(os.environ.get("ROBOSYN_WORK_ROOT", "/workspace/challenge/robosyn-groot")))
    workspace.initialize()
    print(run_closed_loop_evaluation(workspace, args.task, smoke=args.smoke))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
