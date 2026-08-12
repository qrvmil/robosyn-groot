# Hugging Face Access Blocker

- Checked: 2026-08-12T13:12:00Z
- `HF_TOKEN` present in process: no
- Hugging Face CLI authenticated: no (`hf auth whoami` returned `Not logged in`)
- Gated model access: not verified
- Required models: `nvidia/Cosmos-Reason2-2B`, `nvidia/GR00T-N1.7-3B`

Safe next action: expose `HF_TOKEN` only in the process environment or authenticate interactively with `uv run hf auth login`. Never write the token to this workspace.
