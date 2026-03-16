# Environment setup

This project requires a Python virtual environment. The repository previously contained archived virtual environments which are now removed from source control.

Quick start (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-test.txt
```

For full runtime (optional, heavy):

```powershell
python -m pip install -r requirements-runtime.txt
# If using PaddleOCR on Windows, follow its install instructions and consider
# installing with `--no-deps` for binary compatibility as documented.
```

If you need to recreate the environment on Linux/macOS, replace the activation step accordingly.

Keep virtual environments out of the repository; they are listed in `.gitignore`.
