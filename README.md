# Imagewoof CNN

A custom sequential PyTorch CNN for classifying ten dog breeds from the instructor-provided Imagewoof dataset. Develop and train locally in VS Code; use this repository to share code and reproducible setup instructions.

## Project files

| File or folder | Purpose |
| --- | --- |
| `submission.py` | CNN architecture, official class order, `load_model`, and `predict` |
| `train.py` | Training, validation, best-checkpoint saving, and inference verification |
| `check_environment.py` | Package imports and a small GPU forward/backward check |
| `requirements-windows-lock.txt` | Tested, pinned Windows NVIDIA environment |
| `requirements-gpu.txt` | GPU package pair plus unpinned supporting dependencies |
| `requirements.txt` | Supporting dependencies without PyTorch |
| `data/` | Local training/validation images and class mapping; ignored by Git |
| `runs/` | Local checkpoints, metrics, and source snapshots; ignored by Git |
| `.venv/` | Local Python environment; ignored by Git |

## First-time setup for teammates

These instructions target **Windows, PowerShell, Python 3.13, and a compatible NVIDIA GPU**. The setup was tested on an RTX 3080 with PyTorch 2.11.0 and CUDA 12.8 wheels. Other operating systems or GPU types need an appropriate [PyTorch installation](https://pytorch.org/get-started/locally/) instead of this Windows lockfile.

### 1. Install prerequisites and clone

Install [Git](https://git-scm.com/downloads), [VS Code](https://code.visualstudio.com/), [uv](https://docs.astral.sh/uv/getting-started/installation/), and a current NVIDIA GPU driver. Install Microsoft's Python extension in VS Code. Accept the repository collaborator invitation before cloning.

Run these commands from the directory where you want to keep the project:

```powershell
git clone https://github.com/Kngo28/Imagewoof-CNN.git
cd Imagewoof-CNN
```

Open the cloned folder using **VS Code > File > Open Folder**, then open **Terminal > New Terminal**. Run all remaining commands from this project folder. If you already have a local copy, open that folder instead of cloning again.

### 2. Create the environment and verify the GPU

```powershell
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-windows-lock.txt
.\.venv\Scripts\python.exe check_environment.py --require-cuda
```

The check should print your GPU name and `PASS`. The GPU packages are a large download; installation only needs to be repeated when rebuilding the environment or updating dependencies.

Press **Ctrl+Shift+P > Python: Select Interpreter** and select `.venv\Scripts\python.exe`. The commands here use that interpreter explicitly, so environment activation is unnecessary.

### 3. Extract the course dataset

Get `imagewoof_train.zip` and `imagewoof_val.zip` from the instructor or the team's private dataset folder. If both are in your Windows Downloads folder, run:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\imagewoof_train.zip" -DestinationPath data -Force
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\imagewoof_val.zip" -DestinationPath data -Force
```

Change the archive paths if you saved them elsewhere. Both archives contain the same `classes.txt`; `-Force` permits that shared file to be extracted again. The resulting layout must be:

```text
data/
  classes.txt
  train/
    australian_terrier/
    ...
  val/
    australian_terrier/
    ...
```

There should be **9,065 training images** and **1,295 validation images**, each RGB and 128 x 128. The training script validates the counts, dimensions, and official class mapping before training. Do not add an extra `train/train` or `val/val` folder level.

Use only the supplied course data and preserve its splits. The public Imagewoof dataset must not be substituted. Labels follow `classes.txt`, not alphabetical folder order.

## Run training

First run a one-epoch rehearsal:

```powershell
.\.venv\Scripts\python.exe -u train.py --data data --output runs/local_check_01 --epochs 1
```

After it finishes successfully, start a full baseline run:

```powershell
.\.venv\Scripts\python.exe -u train.py --data data --output runs/local_baseline_01 --epochs 30 --patience 7 --batch-size 64 --seed 42
```

- Each run needs a **new output folder**. Use names such as `runs/nev_baseline_02` to identify the teammate and experiment.
- Training saves the best validation checkpoint and stops after seven epochs without improved validation accuracy, or after the epoch limit.
- If GPU memory runs out, try `--batch-size 32` with a new output folder.
- Keep the PC awake and the training terminal running. Files persist after interruption, but the current script **does not resume training**; a new invocation starts from scratch.
- Fixed seeds help reproducibility, but results may differ across hardware and package versions.

For normal use, open the project and run the training command. You do not need to recreate the environment or extract the dataset again.

## Results and model files

Each run saves:

| File | Contents |
| --- | --- |
| `model.pth` | Weights with the best validation accuracy |
| `submission.py`, `train.py` | Source snapshots for that run |
| `config.json` | Run settings, seed, package versions, and device details |
| `history.csv` | Per-epoch losses, accuracies, and learning rate |
| `verification.json` | Final prediction-interface and CPU inference checks, written after successful verification |

Keep each checkpoint with its matching `submission.py`. Back up important run folders to the team's private file storage; ignored local files are not backed up by GitHub. To share experiment summaries through Git, copy selected small metrics or plots into a `results/` folder and record the source commit and checkpoint location.

## Team workflow

GitHub stores source, dependency files, documentation, and selected small results. `.gitignore` excludes environments, datasets, run folders, Python caches, ZIP archives, model weights, and `.env` files. Each teammate installs their own environment and keeps their own dataset copy.

Start a change from an up-to-date `main` with a clean working tree:

```powershell
git switch main
git pull --ff-only origin main
git switch -c your-name/change-description
```

After editing:

```powershell
git add .
git diff --cached --stat
git status
git commit -m "Describe the change"
git push -u origin HEAD
```

Review the staged files before committing, then open a pull request on GitHub for teammate review. `git add .` respects `.gitignore`, although ignore rules do not remove files already tracked by Git. Coordinate dependency changes and update the environment instructions/version pins when needed.

## Assignment submission

Use a custom sequential CNN built from allowed standard layers. Do not use pretrained weights, external data, skip connections, attention, or hyperparameter/architecture-search tools.

The self-contained `submission.py` exposes:

```python
load_model(path="model.pth")  # Load the matching architecture/weights in eval mode.
predict(model, images)       # Raw uint8 RGB (N,128,128,3) -> integer class IDs (N,).
```

Normalization and class order are embedded in the inference file. Importing it does not start training. At the end of training, the script checks validation accuracy through `predict` and performs a CPU inference check. Rehearse the final files in a fresh environment before submission.

Package only the chosen run's **`submission.py` and `model.pth`** at the root of `GroupX_Assignment.zip`, replacing X with the group number. Include each teammate's name and ID in the submission comment. The specification lists **November 7, 2026, at 11:59 pm**; confirm the course timezone and complete both file submission and the live Teams evaluation before the deadline. The version evaluated live is final.
