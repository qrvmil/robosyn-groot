# Unified GR00T README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create one root README that explains the completed preparation, exact checkpoint and parameters, launch/resume workflow, and safe multi-task dataset extension.

**Architecture:** The root `README.md` is the human entry point and links to existing machine-generated evidence rather than duplicating large logs. Every command uses the current absolute workspace paths and every model/data identifier comes from the committed launch manifest.

**Tech Stack:** Markdown, Bash commands, Hugging Face Hub cache, Isaac-GR00T, LeRobot v2.1.

## Global Constraints

- Do not launch the 2,000-step training run.
- Distinguish the NVIDIA base checkpoint from the local 500-step tiny validation checkpoint.
- Describe multiple datasets using Linux `os.pathsep` (`:`) exactly as supported by the pinned launcher.
- State the all-zero gripper-action limitation prominently.

---

### Task 1: Write and verify the root README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: `runs/click_bell_sim_baseline_2k/launch_manifest.json`, launcher scripts, modality config, and committed reports.
- Produces: a single human-readable entry point for launch and dataset extension.

- [x] **Step 1:** Write sections for status, completed work, checkpoints, model parameters, inputs/outputs, launch, monitoring, resume, multi-task data, validation, and troubleshooting.
- [x] **Step 2:** Verify all referenced local files exist and all Bash scripts pass `bash -n`.
- [x] **Step 3:** Scan for placeholders and contradictions with `launch_manifest.json` and `command.sh`.
- [x] **Step 4:** Run the existing test suite and commit only the documentation change and this plan.
