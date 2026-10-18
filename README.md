# AER1515: Affect Perception Project for AI Tutors

Using OpenFace3.0 for feature extraction, Perspective-n-Point (PnP) for 2D to 3D mapping, Kalman smoothing, and LSTM classification based on the DAiSEE dataset.

## Environment
- macOS (Apple Silicon, M1), python 3.11 venv managed by [uv](https://docs.astral.sh/uv/)
- OpenFace3.0 @ commit https://github.com/CMU-MultiComp-Lab/OpenFace-3.0.git
- All setup steps below are macOS specific (`sed -i ''`, BSD `stat`)

## Repo Layout
- `src` - my code (feature extraction, PnP, Kalman, LSTM)
- `scripts/` - setup helpers (weight download, macOS patch)
- `OpenFace-3.0` - cloned repo **gitignored**
- `.venv/`, `data/`, `results/` - gitignored (env, DAiSEE, outputs)

## Setup from scratch
```bash 
## 0. Prereqs: git
## Install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh #restart terminal after

## 1. Get the code
git clone https://github.com/pauldsun/affect-detection.git AffectProject
cd AffectProject
git clone https://github.com/CMU-MultiComp-Lab/OpenFace-3.0.git

## 2. Environment
uv python install 3.11
uv venv --python 3.11 --seed .venv
source .venv/bin/activate  # for every session
python --version           # sanity check (expect 3.11.x)

## 3. Dependencies
uv pip install -r OpenFace-3.0/requirements.txt
uv pip install openface-test

## 4. Model weights + macOS patch
./scripts/download_weights.sh      # ~460 MB, verifies sizes
./scripts/patch_openface.sh        # fixes hardcoded /work path (see Problems)

## 5. Verify (run from inside OpenFace-3.0 due to weights)
cd OpenFace-3.0
openface detect images/0.jpg --output-dir ../results --device cpu
## results/face_analysis_0.csv appears with landmarks/gaze/AU/emotion columns
```

## Problems:
1. `openface download` skips if `weights/` already exists, but `weights/` always comes with a fresh clone of OpenFace-3.0. Use `scripts/download_weights.sh` instead.
2. The pip package hardcodes the author's Docker path (`/work/...`) and crashes on macOS with `[Errno 30] Read-only file system`. To fix this, use 'scripts/patch_openface.sh'. **after every venv rebuild**
3. The CLI defaults to `--device cuda`. On Mac, always use `--device cpu`.
4. The OpenFace-3.0 `demo.py` and `demo2.py` rely on empty directories, thus they are unusable.

## Status Log:
- 2026-10-17: Environment Setup + OpenFace working. Next: visualization script and feature extraction script.

## Citations:
```bibtex
@article{hu2025openface,
  title={OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis},
  author={Hu, Jiewen and Mathur, Leena and Liang, Paul Pu and Morency, Louis-Philippe},
  journal={arXiv preprint arXiv:2506.02891},
  year={2025}
}
```