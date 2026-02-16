# MLForEverything
## Prerequisites

### Installing UV Package Manager

#### What is UV?

UV is an extremely fast Python package installer and resolver, written in Rust. It's designed as a drop-in replacement for pip and pip-tools, offering significant performance improvements for package installation and dependency resolution.

#### Advantages of Using UV

- **Speed**: 10-100x faster than pip for package installation and dependency resolution
- **Reliable**: Consistent dependency resolution with a robust resolver
- **Drop-in Replacement**: Compatible with pip commands and workflows
- **Disk Space Efficient**: Uses a global cache to avoid redundant downloads
- **Better Error Messages**: Provides clearer feedback when dependency conflicts occur
- **Modern**: Built with modern Rust tooling for performance and reliability

#### Installation Instructions

##### macOS

Install UV using Homebrew:
```bash
brew install uv
```

Or using curl:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using pip:
```bash
pip install uv
```

##### Linux (Ubuntu/Debian)

Using curl:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using pip:
```bash
pip install uv
```

Or using pipx (recommended for system-wide installation):
```bash
pipx install uv
```

##### Windows

Using PowerShell:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or using pip:
```powershell
pip install uv
```

Or using Scoop:
```powershell
scoop install uv
```

#### Verify Installation

After installation, verify that UV is installed correctly:
```bash
uv --version
```

#### Basic UV Usage

Replace `pip` with `uv pip` in your commands:
```bash
# Install a package
uv pip install package-name

# Install from requirements.txt
uv pip install -r requirements.txt

# Install in a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install package-name
```

---

## Installing Jupyter Notebook with UV

### Step 1: Create a Virtual Environment (Recommended)

First, create a virtual environment using UV:
```bash
uv venv
```

This creates a `.venv` directory in your project.

### Step 2: Activate the Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```powershell
.venv\Scripts\activate
```

### Step 3: Install Jupyter Notebook

Install Jupyter Notebook using UV:
```bash
uv pip install notebook
```

Or install JupyterLab (modern interface):
```bash
uv pip install jupyterlab
```

**Alternative: Install All Dependencies at Once**

If you have a `requirements.txt` file with all your project dependencies, you can install everything in one command:
```bash
uv pip install -r requirements.txt
```

This will install Jupyter, data science libraries, deep learning frameworks, and visualization tools all together.

### Step 4: Create a Jupyter Kernel (Optional but Recommended)

To use your virtual environment as a kernel in Jupyter Notebook, install and register it:

```bash
# Install ipykernel
uv pip install ipykernel

# Register the kernel with Jupyter
python -m ipykernel install --user --name=mlforeverything --display-name="Python (MLForEverything)"
```

This creates a kernel named "mlforeverything" that will appear in Jupyter's kernel selection menu.

**Parameters explained:**
- `--user`: Installs the kernel for the current user only
- `--name`: Internal name for the kernel (used in kernel specs)
- `--display-name`: Name shown in Jupyter's interface

### Step 5: Launch Jupyter Notebook

Start Jupyter Notebook:
```bash
jupyter notebook
```

Or start JupyterLab:
```bash
jupyter lab
```

This will open Jupyter in your default web browser at `http://localhost:8888`.

When creating a new notebook, select "Python (MLForEverything)" from the kernel menu.

### Managing Kernels

**List all installed kernels:**
```bash
jupyter kernelspec list
```

**Remove a kernel:**
```bash
jupyter kernelspec remove mlforeverything
```

**Change to a different kernel in a running notebook:**
Go to `Kernel` → `Change Kernel` in the Jupyter Notebook menu.

### Alternative: Install and Run in One Step

You can also use UV to run Jupyter without explicitly installing it first:
```bash
uv run jupyter notebook
```

This automatically installs Jupyter in an isolated environment and runs it.

### Installing Additional Packages

Once your virtual environment is active, you can install additional packages for your notebooks:

**Option 1: Install from requirements.txt (Recommended)**
```bash
uv pip install -r requirements.txt
```

**Option 2: Install packages individually**
```bash
# Install data science packages
uv pip install numpy pandas matplotlib seaborn scikit-learn

# Install deep learning frameworks
uv pip install torch tensorflow

# Install visualization libraries
uv pip install plotly bokeh altair
```

### Stopping Jupyter

To stop the Jupyter server, press `Ctrl+C` in the terminal where it's running.