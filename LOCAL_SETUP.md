# Local Windows GPU workflow

Open this folder itself in VS Code. The Python interpreter is .venv/Scripts/python.exe.
Use PowerShell commands below, not notebook ! or % prefixes. Install the Microsoft Python
extension if needed, and use Ctrl+Shift+P > Python: Select Interpreter to select the .venv.

## Setup on a teammate's machine

Install Git, uv, and a current NVIDIA driver. Clone the private repository and enter it.
For an NVIDIA GPU supported by the CUDA 12.8 PyTorch build:

```powershell
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-gpu.txt
.\.venv\Scripts\python.exe check_environment.py --require-cuda
```

Once requirements-windows-lock.txt exists, use that instead of requirements-gpu.txt
to reproduce the tested package versions. Other platforms/GPUs may require a different
PyTorch build; see https://pytorch.org/get-started/locally/.

Obtain the two original instructor ZIPs through your private shared dataset folder.
Extract both into data/ so data/train/, data/val/, and data/classes.txt exist.
Do not create a further train/train or val/val nesting level. Keep the original split.

## Train

```powershell
.\.venv\Scripts\python.exe -u train.py --data data --output runs/local_check_01 --epochs 1
.\.venv\Scripts\python.exe -u train.py --data data --output runs/local_baseline_01 --epochs 30 --patience 7 --batch-size 64 --seed 42
```

Run the short rehearsal first. Each command needs a new output folder. A CUDA
out-of-memory error can be addressed by trying batch size 32 in a new run.
Keep the computer awake and the training process running. Closing VS Code's training
terminal or shutting down interrupts training. Files persist after interruption, but
this script does not resume training: running it again starts fresh.

The best model and matching source snapshot are saved in the run folder. Copy existing
Colab run folders from Drive into runs/colab_baseline_01 to preserve earlier results.
Always keep a checkpoint with the submission.py that produced it.

## Collaborate

Commit source, setup instructions, dependency files, and small selected results to GitHub.
Do not commit .venv, datasets, ZIPs, or model weights; .gitignore already excludes them.
Teammates create their own environments and data folders once, then use git pull for code
updates. Share weights and data through a private Drive folder or another agreed artifact
store. Local files persisting does not make them backed up; keep a copy of important runs.

```powershell
git status
git add train.py submission.py check_environment.py requirements.txt requirements-gpu.txt requirements-windows-lock.txt LOCAL_SETUP.md README.md .gitignore .vscode/settings.json
git commit -m "Add local GPU training setup"
git push
```

No Colab mounting or extraction cells are needed for ordinary local training.
