#!/bin/bash
# Run this in Git Bash from repo root to fix the push (remove large files from tracking)

set -e
cd "$(dirname "$0")"

echo "1. Removing __pycache__ from git index..."
git rm -r --cached FacialKeypointsDetection/src/__pycache__ 2>/dev/null || true
git rm -r --cached HumanActionRecognition/src/__pycache__ 2>/dev/null || true

echo "2. Removing large files from git index..."
git rm --cached FacialKeypointsDetection/data/raw/training.csv 2>/dev/null || true
git rm --cached FacialKeypointsDetection/data/raw/test.csv 2>/dev/null || true
git rm --cached FacialKeypointsDetection/models/final_model.pkl 2>/dev/null || true

echo "3. Adding .gitignore..."
git add .gitignore

echo "4. Amending the last commit..."
git commit --amend -m "Add .gitignore and stop tracking large files and cache"

echo "Done. Now run: git push origin main"
echo "(If you had already pushed the previous commit, use: git push origin main --force-with-lease)"
