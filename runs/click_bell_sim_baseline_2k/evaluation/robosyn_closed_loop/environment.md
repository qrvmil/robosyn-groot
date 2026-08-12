# RoboSyn closed-loop evaluation environment

- Captured (UTC): `2026-08-12T22:38:44Z`
- GPU: `NVIDIA A100 80GB PCIe`
- NVIDIA driver: `580.126.20`
- GPU memory: `81920 MiB`
- Simulator Python: `3.11.15`
- EmbodiChain commit: `3d1827fa968ab3bc7e00f9b8ed8ac589683259f9`
- RoboSyn adapter commit: `32e717cc60c350469b2af68752da71bd8b98f2e0`
- Isaac-GR00T commit: `376ba890cff8c9de64d71d982772a9c36185fdd7`
- Checkpoint: `runs/click_bell_sim_baseline_2k/checkpoints/click_bell_sim_baseline_2k/checkpoint-2000`
- Task/setting: `click_bell clear`
- Execution horizon: `13`
- System runtime additions: `libopengl0=1.7.0-1build1`, `libglu1-mesa=9.0.2-1.1build1`, `libsm6=2:1.2.3-1build3`, `libice6=2:1.0.10-1build3`

The GR00T model runs in its existing Python 3.12 environment. RoboSyn and
EmbodiChain run in `/workspace/challenge/robosyn-groot/.venvs/robosyn` and
communicate with GR00T over a localhost ZeroMQ socket.
