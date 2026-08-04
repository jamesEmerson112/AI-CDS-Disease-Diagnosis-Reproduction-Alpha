# James's WSL Development Setup Guide

## Current Environment

**Date Created:** November 21, 2025  
**WSL Distribution:** Linux 6.6  
**Default Shell:** /bin/sh

## Project Location

### Windows Path
```
C:\Users\voan2\Documents\GitHub\AI-CDS-Disease-Diagnosis-Reproduction
```

### WSL Path (Current)
```
/mnt/host/c/Users/voan2/Documents/GitHub/AI-CDS-Disease-Diagnosis-Reproduction
```

### Important Note on Path Structure
- In this WSL setup, Windows drives are mounted under `/mnt/host/` (not the typical `/mnt/`)
- Windows `C:\` → WSL `/mnt/host/c/`
- Windows `D:\` → WSL `/mnt/host/d/`

## Current Setup: Option 1 (Working from Windows Filesystem)

### Overview
Currently working directly from the Windows filesystem via WSL's `/mnt/host/c/` mount point.

### ✅ Pros
1. **No Duplication** - Don't waste 20.9GB copying the BioSentVec model file
2. **Simple Setup** - No migration needed, works immediately
3. **Seamless Sync** - Files automatically accessible from both Windows and WSL
4. **Windows Access** - Can edit with Windows apps (Notepad++, PyCharm, etc.)
5. **Backups/Git** - Windows backup solutions continue working
6. **Easy to Find** - Files remain in familiar Windows location

### ❌ Cons
1. **Slower File I/O** - Cross-filesystem operations are 2-10x slower
   - Opening/saving files takes longer
   - Git commands slower (status, commit, pull, etc.)
   - File searches/greps slower
   - Installing Python packages slower
2. **Potential File Issues**
   - Line ending problems (Windows CRLF vs Linux LF)
   - Case sensitivity differences
   - Permission/ownership quirks
3. **Performance Impact**
   - VSCode file watching/indexing slower
   - Linting/autocomplete might lag
   - Test execution slower if creating temp files
4. **Not Best Practice** - Microsoft recommends WSL filesystem for dev work

### Real Impact for THIS Project
- **Model loading (20.9GB)**: Same speed - only loads once at startup
- **Python execution**: Similar speed
- **Code editing**: Slightly slower but likely not noticeable
- **Git operations**: Noticeably slower
- **pip install**: Slower package installation

## Project Files

### Large Files
- `BioSentVec_PubMed_MIMICIII-bigram_d700.bin` - **20.9 GB** - BioSentVec model (excluded from Git)

### Key Files
- `CS2V.py` - Main script
- `requirements.txt` - Python dependencies
- `setup.py` - Project setup
- `util_cy.c` / `util_cy.cp39-win_amd64.pyd` - Cython compiled modules

### Directories
- `Dataset/` - Training/test data in 10 folds (Fold0-Fold9)
- `entity/` - Data models (Admission, Symptom, etc.)
- `utils/` - Utility modules
- `venv/` - Virtual environment
- `build/` - Build artifacts

## Alternative Option (Not Currently Used)

### Option 2: Hybrid Approach - Copy to WSL Filesystem

**If performance becomes an issue**, consider this hybrid setup:

1. Copy project to WSL native filesystem (e.g., `/root/projects/`)
2. Delete the copied 20.9GB .bin file
3. Create symlink pointing back to Windows version

```bash
# Commands to switch to hybrid setup
cp -r /mnt/host/c/Users/voan2/Documents/GitHub/AI-CDS-Disease-Diagnosis-Reproduction /root/projects/
rm /root/projects/AI-CDS-Disease-Diagnosis-Reproduction/BioSentVec_PubMed_MIMICIII-bigram_d700.bin
ln -s /mnt/host/c/Users/voan2/Documents/GitHub/AI-CDS-Disease-Diagnosis-Reproduction/BioSentVec_PubMed_MIMICIII-bigram_d700.bin \
      /root/projects/AI-CDS-Disease-Diagnosis-Reproduction/
cd /root/projects/AI-CDS-Disease-Diagnosis-Reproduction
```

**Benefits:**
- ⚡ 5-10x faster file operations
- Native Linux performance for Git, package installs, etc.
- Only one copy of the 20.9GB model (via symlink)

## Useful Commands

### Check Current Location
```bash
pwd
# Should show: /mnt/host/c/Users/voan2/Documents/GitHub/AI-CDS-Disease-Diagnosis-Reproduction
```

### Check File Sizes
```bash
du -h BioSentVec_PubMed_MIMICIII-bigram_d700.bin
ls -lh *.bin
```

### Navigate Between Windows and WSL
```bash
# From WSL to Windows drives
cd /mnt/host/c/Users/voan2/Documents
cd /mnt/host/d/  # D: drive

# From Windows, open WSL in specific directory
# In PowerShell/CMD:
cd C:\Users\voan2\Documents\GitHub\AI-CDS-Disease-Diagnosis-Reproduction
wsl
```

### Open VSCode from WSL
```bash
code .
```

## VSCode Configuration

### Current Workspace
- Working directory: `/mnt/host/c/Users/voan2/Documents/GitHub/AI-CDS-Disease-Diagnosis-Reproduction`
- VSCode connected to: WSL (Linux 6.6)
- Cline extension: Active

### If Terminal Fails to Launch
If you see error: "Starting directory (cwd) does not exist"
- Check VSCode workspace settings
- Ensure the directory path exists
- Reopen folder: **File → Open Folder** → navigate to project

## Notes

- This setup allows working across Windows and WSL seamlessly
- Current performance is acceptable for ML/AI development workflow
- The 20.9GB model file is already excluded in `.gitignore` (*.bin pattern)
- If Git/file operations become too slow, consider switching to Option 2 (hybrid)
- Remember: This doc is excluded from Git via `.gitignore`

## Troubleshooting

### Issue: Slow Git Operations
**Solution:** Consider moving to WSL native filesystem (Option 2)

### Issue: Line Ending Problems
**Solution:** Git should auto-convert, but ensure `.gitattributes` is configured properly

### Issue: Permission Errors
**Solution:** Check file permissions with `ls -l`, adjust with `chmod` if needed

### Issue: Can't Execute Commands
**Solution:** Ensure shell exists: `which sh` or `which bash`

---

**Last Updated:** November 21, 2025  
**Status:** ✅ Working from Windows filesystem (Option 1)
