# Install or update AssetFurnace on Windows with an NVIDIA GPU. LIMITED TESTING: if you run it,
# please tell us how it went.
#
#   git clone https://github.com/matteodante/AssetFurnace.git; cd AssetFurnace; .\install.ps1
#
# Re-running it from the clone is how you update that clone. It never downloads model
# weights; those are chosen, sized and agreed to in the viewer's Setup & Status page.
#
# Options: set these before running, e.g. $env:I3D_DIR = "D:\lab"
#   I3D_DIR    where to install (default: the clone this script is in, else $HOME\image-to-3dlab)
#   I3D_REF    a release tag or branch (default: the repository's newest vX.Y.Z release, else main)
#   I3D_REPO   clone from here instead of github.com/matteodante/AssetFurnace
#   I3D_YES    "1" for no questions (scripts and agents)

$ErrorActionPreference = "Stop"
$Repo = if ($env:I3D_REPO) { $env:I3D_REPO } else { "https://github.com/matteodante/AssetFurnace.git" }
# Run as .\install.ps1 from a clone, it updates that clone rather than making another one.
$Here = if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot ".git"))) { $PSScriptRoot } else { $null }
$Dir  = if ($env:I3D_DIR) { $env:I3D_DIR } elseif ($Here) { $Here } else { Join-Path $HOME "image-to-3dlab" }
$Ref  = $env:I3D_REF
$Yes  = $env:I3D_YES -eq "1"

function Say($m) { Write-Host "[install] $m" -ForegroundColor Cyan }
function Die($m) { Write-Host "[install] $m" -ForegroundColor Red; exit 1 }
function Confirm-Step($question) {
    if ($Yes) { return }
    if (-not [Environment]::UserInteractive) { Die "$question Nobody to ask, so stopping. Set `$env:I3D_YES = '1' to go ahead." }
    $answer = Read-Host "$question [y/N]"
    if ($answer -notmatch '^(y|yes)$') { Die "Stopped. Nothing was changed." }
}

# 1. This machine
if (-not [Environment]::Is64BitOperatingSystem) { Die "The lab needs 64-bit Windows." }
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$HasGpu = [bool]($smi -and ((& nvidia-smi -L) -match "GPU"))
if ($HasGpu) { Say "Machine: Windows with an NVIDIA GPU" }
else { Say "No NVIDIA GPU found (nvidia-smi lists none). The viewer will install, but no generation route will run here until one is." }

# 2. Tools
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Die "git is missing. Install it from https://git-scm.com/download/win, then run this again."
}
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Say "uv (Astral's Python installer, ~40 MB) is needed to set up Python 3.11."
    Confirm-Step "Install uv from astral.sh?"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$HOME\.local\bin;$env:Path"
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) { Die "uv installed, but this window cannot see it yet. Open a new PowerShell window and run this again." }
}

# 3. Code: clone, or update an existing install
if (Test-Path (Join-Path $Dir ".git")) {
    Say "Updating the install in $Dir"
    if (git -C $Dir status --porcelain --untracked-files=no) {
        Die "$Dir has local edits to tracked files. Commit or stash them, then run this again."
    }
    git -C $Dir fetch --tags --force --quiet $Repo "+refs/heads/*:refs/remotes/origin/*"
} elseif ((Test-Path $Dir) -and (Get-ChildItem $Dir -Force | Select-Object -First 1)) {
    Die "$Dir exists and is not an image-to-3dlab install. Set `$env:I3D_DIR to another place."
} else {
    Say "Installing into $Dir"
    git clone --quiet $Repo $Dir
}
if (-not $Ref) {
    # The newest release of the repository being installed. Not `git tag --list`: a clone
    # that once fetched another remote's tags (the upstream's v0.4.3) keeps them, and an
    # update would go back to that release.
    $Ref = git ls-remote --tags --refs $Repo "v[0-9]*.[0-9]*.[0-9]*" |
        ForEach-Object { ($_ -split "refs/tags/")[1] } |
        Where-Object { $_ -match '^v\d+\.\d+\.\d+$' } |
        Sort-Object { [version]$_.Substring(1) } -Descending | Select-Object -First 1
    if (-not $Ref) { $Ref = "main" }
}
Say "Version: $Ref"
git -C $Dir rev-parse --verify --quiet "refs/remotes/origin/$Ref" *> $null
if ($LASTEXITCODE -eq 0) { git -C $Dir checkout --quiet -B $Ref "origin/$Ref" }
else { git -C $Dir checkout --quiet $Ref }

# 4. Python
Say "Setting up Python 3.11 and the viewer's packages (a few GB with an NVIDIA GPU, mostly PyTorch)"
& uv venv --quiet --allow-existing --python 3.11 (Join-Path $Dir ".venv")
$Py = Join-Path $Dir ".venv\Scripts\python.exe"

# The lock pins every package to versions tested together, as install.sh installs them;
# requirements.txt only bounds them, so installing from it gave each install whatever
# PyPI had that day.
$Reqs = Join-Path $Dir "requirements.lock"
if (-not (Test-Path $Reqs)) { $Reqs = Join-Path $Dir "requirements.txt" }

# The version a requirements file pins for a package; "" when it only bounds it, as
# requirements.txt does in a checkout older than the lock.
function Get-Pin([string]$Path, [string]$Name) {
    $found = Select-String -Path $Path -Pattern "^$Name==([^ ;]+)" | Select-Object -First 1
    if ($found) { $found.Matches[0].Groups[1].Value } else { "" }
}

$Install = $Reqs
$TorchVersion = Get-Pin $Reqs "torch"
$VisionVersion = Get-Pin $Reqs "torchvision"

# PyPI only has CPU-only PyTorch for Windows, so on an NVIDIA machine install the CUDA
# build from PyTorch's own index first, at the lock's versions: the lock then finds torch
# already there (2.x+cu1yy satisfies ==2.x) instead of replacing it with the CPU wheel.
# image_to_3dlab.host picks the build that matches the driver (tested in tests/test_host.py).
# An update also replaces a CPU-only torch left by an older installer.
if ($HasGpu) {
    Push-Location $Dir
    $TorchIndex = (& $Py -m image_to_3dlab.host torch-index | Out-String).Trim()
    & $Py -m image_to_3dlab.host torch-has-cuda
    $TorchHasCuda = ($LASTEXITCODE -eq 0)
    Pop-Location
    if (-not $TorchIndex) {
        Say "Your NVIDIA driver is too old for PyTorch's CUDA builds (they need CUDA 12.8 or newer). Update the driver, then run this again. Until then PyTorch will use the CPU."
    } else {
        # An index that stops at an older torch than the lock's (cu128 has up to 2.11.0, the
        # lock's is on cu130 only) has a lock of its own, compiled from the same requirements
        # with that torch: requirements-cu128.lock.
        $Tag = ($TorchIndex -split "/")[-1]
        $Own = Join-Path $Dir "requirements-$Tag.lock"
        if (Test-Path $Own) {
            $Install = $Own
            $TorchVersion = Get-Pin $Own "torch"
            $VisionVersion = Get-Pin $Own "torchvision"
            Say "Your driver runs PyTorch's $Tag build, which goes up to torch $($TorchVersion): installing that (a driver for CUDA 13 gets the newest)."
        }
        $Torch = if ($TorchVersion) { @("torch==$TorchVersion", "torchvision==$VisionVersion") } else { @("torch", "torchvision") }
        if (-not $TorchHasCuda) {
            Say "Installing PyTorch with CUDA from $TorchIndex"
            & uv pip install --quiet --python $Py --reinstall-package torch --reinstall-package torchvision --index-url $TorchIndex @Torch
        } else {
            # the locked version, if this install has another one
            & uv pip install --quiet --python $Py --index-url $TorchIndex @Torch
        }
        if ($LASTEXITCODE -ne 0) { Die "PyTorch ($($Torch -join ', ')) with CUDA could not be installed from $TorchIndex. Nothing else was changed in the packages." }
    }
}
# (falls back to "requirements.txt" only in a checkout older than the lock; the driver's
# own lock when its PyTorch index has another torch)
& uv pip install --quiet --python $Py -r $Install

# 5. Blender: Finish needs it, and it is the user's to install. Same finder Finish uses.
if (Test-Path $Py) {
    Push-Location $Dir
    & $Py -c "import sys; from image_to_3dlab.blender import find_blender; sys.exit(0 if find_blender() else 1)" 2>$null
    $HasBlender = ($LASTEXITCODE -eq 0)
    Pop-Location
    if (-not $HasBlender) {
        Say "Blender was not found. Finish (the low-poly clean-up) needs Blender 4.2 or newer: https://www.blender.org/download/"
    }
}

# 6. Done
Say "Done. Start the lab with:"
Write-Host ""
Write-Host "    cd `"$Dir`"; .venv\Scripts\python.exe viewer\serve.py"
Write-Host ""
Say "Then open Setup & Status to choose what to install. Nothing large downloads until you say so there."
Say "To update later, run this installer again, then restart the viewer."
