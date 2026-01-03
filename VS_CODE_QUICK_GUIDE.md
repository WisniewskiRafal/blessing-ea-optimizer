# VS Code GitHub Guide - QUICK VERSION

## Step 1: Open Terminal in VS Code
Press: **Ctrl + `** (backtick)

## Step 2: Initialize Git
```bash
git init
git status
```

## Step 3: Check Files
Look at the output. Should see:
- ✅ README.md, LICENSE, .py files
- ❌ NO data/results/, NO .set files

## Step 4: Add and Commit
```bash
git add .
git commit -m "Initial commit: Blessing EA Optimizer v3.0"
```

## Step 5: Create GitHub Repo
Go to: https://github.com/new
- Name: blessing-ea-optimizer
- Public
- DO NOT add README/License
- Click Create

## Step 6: Push
Replace YOUR-USERNAME with your GitHub username:
```bash
git remote add origin https://github.com/YOUR-USERNAME/blessing-ea-optimizer.git
git branch -M main
git push -u origin main
```

When prompted:
- Username: your GitHub username
- Password: Personal Access Token (get from https://github.com/settings/tokens)

## DONE! 🎉
Visit: https://github.com/YOUR-USERNAME/blessing-ea-optimizer
