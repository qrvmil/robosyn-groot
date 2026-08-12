# Authoritative Gated File Access Result

- Checked: 2026-08-12T13:36:00Z
- Hugging Face CLI account: `qrvmil`
- Authentication: passed
- Metadata lookup for both required repositories: passed
- Download of `nvidia/Cosmos-Reason2-2B/config.json`: failed with HTTP 403
- Gate A: failed

The earlier `gated_model_access.txt` records metadata visibility only and does not prove gated file access. The authoritative evidence is `runs/so100_smoke/stdout.log`: GR00T model initialization attempted the actual file download, and Hugging Face reported that account `qrvmil` is not in the authorized list.

Required action: request and receive access at <https://huggingface.co/nvidia/Cosmos-Reason2-2B>. Authentication alone is insufficient. Never place the token in this repository or chat.
