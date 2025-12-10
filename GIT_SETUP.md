# Git Setup Guide

## Current Status

❌ **Git NOT initialized**

## Quick Setup Commands

### Option 1: Initialize Git Now

```bash
cd /media/mohi/Mohi-M11/Arman/stage1_cleanup

# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: Stage 1 Animation Cleanup App"

# Check status
git status
```

### Option 2: Initialize with GitHub

```bash
# 1. Initialize
git init

# 2. Add files
git add .

# 3. First commit
git commit -m "Initial commit: Stage 1 Animation Cleanup App"

# 4. Add remote (replace with your GitHub repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 5. Push to GitHub
git branch -M main
git push -u origin main
```

## What Will Be Tracked

✅ **Will be tracked:**
- All Python files (.py)
- Configuration files (.yaml, .toml)
- Documentation (.md)
- Requirements (requirements.txt)
- Model checkpoint (epoch_014.pth) - if you want

❌ **Will be ignored** (via .gitignore):
- `__pycache__/` folders
- Virtual environments
- IDE files
- Output folders (test_output/, outputs/)
- Temporary files

## Important: Model Checkpoint

**epoch_014.pth (163MB)** will be tracked by default.

**Options:**
1. **Track it** (recommended for easy deployment)
   - Keep as is
   - Easy to deploy on Streamlit Cloud

2. **Don't track it** (if repo gets too large)
   - Add to .gitignore:
     ```bash
     echo "*.pth" >> .gitignore
     ```
   - Upload separately to cloud storage
   - Load from URL in app

## Next Steps After Git Init

1. **Create GitHub Repository:**
   - Go to github.com
   - Click "New repository"
   - Name: `stage1-animation-cleanup`
   - Don't initialize with README (we have one)

2. **Connect and Push:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/stage1-animation-cleanup.git
   git push -u origin main
   ```

3. **Deploy to Streamlit Cloud:**
   - Go to share.streamlit.io
   - Connect GitHub repo
   - Deploy!

## Git Commands Reference

```bash
# Check status
git status

# Add files
git add .
git add specific_file.py

# Commit
git commit -m "Your message"

# Push to GitHub
git push

# Check remote
git remote -v

# View history
git log --oneline
```

---

**Ready to initialize? Run:**
```bash
git init
git add .
git commit -m "Initial commit: Stage 1 Animation Cleanup App"
```

