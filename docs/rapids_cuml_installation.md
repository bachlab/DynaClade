# Install rapids cuml

**Note:** Only possible to use in Linux or WSL2

## Requirements:

### 1. Install WSL2

1. Enable WSL in Windows Features:
   - Open PowerShell as Administrator
   - Run: `dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`
   - Run: `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart`
   - Restart your computer

2. Download and install the WSL2:
   - Open PowerShell as Administrator
   - Run: `wsl --install`

3. Set WSL2 as default:
   - Open PowerShell as Administrator
   - Run: `wsl --set-default-version 2`

4. Verify installation:
   - Open PowerShell
   - Run: `wsl --list --verbose`
   - Should show Ubuntu running on version 2

### 2. Install Miniconda in WSL2

1. Download Miniconda installer:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

2. Make the installer executable:
   ```bash
   chmod +x Miniconda3-latest-Linux-x86_64.sh
   ```

3. Run the installer:
   ```bash
   ./Miniconda3-latest-Linux-x86_64.sh
   ```
   - Follow the prompts to complete installation
   - Accept the license agreement
   - Confirm installation location
   - Allow the installer to initialize Miniconda3

4. Restart your terminal or run:
   ```bash
   source ~/.bashrc
   ```

5. Verify installation:
   ```bash
   conda --version
   ```


### 3. Create environment with rapids including cuml, run:

1. Create environment with rapids including cuml:
   ```bash
   conda create --solver=libmamba -n rapids-23.12 -c rapidsai -c conda-forge -c nvidia rapids=23.12 python=3.10 cuda-version=12
   ```

**Note:** modify cuda-version according to suitable system version

### 4. Activate environment:

1. Activate environment:
   ```bash
   conda activate rapids-23.12
   ```
