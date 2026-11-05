# HW2 Imagewoof team setup

Use a private GitHub repository as the shared source of truth and Colab GPU sessions for training. Each teammate has a separate runtime. This starter contains environment setup only, not an assignment solution.

## 1. Repository owner: publish this folder

Create an empty private GitHub repository, then run these commands inside this folder (replace OWNER and REPO):

```powershell
git init -b main
git add .
git commit -m "Add project environment starter"
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
```

Invite teammates under repository Settings > Collaborators. Use your normal GitHub login or credential manager; never put access tokens in notebooks. The course data and model weights are excluded from Git.

## 2. Agree on the working layout

```text
submission.py         # eventual self-contained model, load_model, predict
train.py              # eventual training entry point; imports the same model
configs/              # small saved experiment configurations
notebooks/            # optional Colab launcher / exploration
results/              # small metrics files and selected plots to commit
data/                 # instructor data, ignored by Git
checkpoints/          # model weights, ignored by Git
```

Only the environment files currently exist. Add these components as you implement the project. Keep reusable code in Python files to reduce notebook merge conflicts.

## 3. Colab: recommended training environment

Create a notebook in Colab and select Runtime > Change runtime type > GPU. Upload the current project source files using Colab's Files panel, or use Colab's authenticated GitHub integration to open a notebook from the private repository. Opening a notebook does not automatically copy other repository files into its runtime; upload those files too. Never embed credentials in a clone URL.

Colab normally provides torch and torchvision. Start with that compatible pair:

```python
%pip install -r requirements.txt
!python check_environment.py --require-cuda
```

If working in a subdirectory, first use `%cd` to enter it. If changing torch/torchvision versions, install a compatible pair from the official PyTorch selector and restart the runtime.

Upload the instructor's train/validation archives and classes.txt to a restricted shared Drive folder. Mount Drive in Colab, copy the archives into runtime-local storage, and extract them there for training. Inspect the actual archive structure before setting paths. Write checkpoints back to Drive regularly: local runtime storage is temporary. Give each teammate/run a unique output directory to prevent overwrites.

After the first successful setup, save the environment record:

```python
!python -m pip freeze > environment-colab.txt
!python check_environment.py --require-cuda > environment-check.txt
```

Download and commit those records. The requirements file is intentionally unpinned until a working environment is verified. Agree on the same torch/torchvision and direct dependency versions, then pin them. A Colab freeze includes unrelated preinstalled packages; treat it as an environment record, not a portable Windows lockfile.

## 4. Optional local development on Windows

Use a dedicated Python 3.12 environment as a conservative team baseline. With uv installed:

```powershell
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe check_environment.py
```

This installs CPU PyTorch for local development and inference checks. For local NVIDIA GPU training, use the install command from https://pytorch.org/get-started/locally/ for your OS and supported CUDA configuration instead. CPU-only local development does not replace the assignment's GPU training setup. Select this folder's .venv interpreter in your editor. No environment activation is required for these commands.

## 5. Validate data before training

- Use only the instructor-provided imagewoof_train.zip and imagewoof_val.zip, plus classes.txt. Do not download the public Imagewoof dataset.
- Verify 9,065 training images and 1,295 validation images; preserve the supplied splits.
- Verify images are RGB and 128 x 128. Inspect a few examples and labels.
- Read classes.txt and explicitly map folder labels into that order. Do not assume torchvision ImageFolder's alphabetical mapping matches it.
- Use random augmentation only for training. Validation and predict must use matching deterministic preprocessing and normalization.
- Record archive hashes so teammates can confirm that they have identical inputs.

## 6. Collaborate and record experiments

Each change gets a branch, a small commit, and a pull request reviewed by a teammate. Pull main before starting. Assign ownership of notebooks to avoid simultaneous edits.

For each run, record the Git commit, seed, architecture, augmentation, optimizer, learning rate/schedule, batch size, epochs, package versions, best validation accuracy, and checkpoint location. Seed Python random, NumPy, PyTorch, and data-loader workers in training code. Fixed seeds improve reproducibility but do not guarantee identical GPU results across platforms.

Manually agree on experiments; the assignment prohibits hyperparameter optimization/search tools. Save the best validation checkpoint with its exact configuration. Reserve time for a fresh-runtime rehearsal rather than spending all remaining time tuning.

## 7. Submission contract and rehearsal

Build a custom sequential CNN from allowed standard layers; no pretrained weights, skip connections, attention, external data, or architecture-search tools.

The final self-contained code file must define:

```python
def load_model(path="model.pth"):
    # Build exact architecture, load weights with portable device mapping,
    # put model in eval mode, return model.
    ...

def predict(model, images):
    # Input: uint8 NumPy RGB array (N, 128, 128, 3).
    # Normalize here; infer in batches without gradients.
    # Return integer NumPy array (N,), class IDs 0-9 in classes.txt order.
    ...
```

Importing the code must not start training, mount Drive, or download data. Because submission contains only the code file and weights, embed the required class mapping/preprocessing constants; do not require extra local config files or classes.txt at inference time.

In a fresh environment, test loading on CPU and GPU and run the exact predict interface over validation images. Confirm its accuracy matches validation during training. Check output shape, dtype, and label range. Package exactly your code file and model.pth as GroupX_Assignment.zip.

The spec states Nov 7, 2026 at 11:59 pm; confirm the course timezone. Both submission and live Teams evaluation must be completed before the deadline. Arrange the meeting early; the evaluated version is final. Include teammate names and IDs in the submission comment as requested.

References: https://pytorch.org/get-started/locally/ and https://research.google.com/colaboratory/faq.html
