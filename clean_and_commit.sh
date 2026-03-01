#!/bin/bash
# Temiz commit: Son commit'i geri al, data/models/cache'i takipten çıkar, .gitignore ile tekrar commit et.
# Git Bash'te proje kökünde çalıştır: bash clean_and_commit.sh

set -e
cd "$(dirname "$0")"

echo "=== 1. Son 3 commit'i geri alıyoruz (değişiklikler durur, sadece commit'ler kalkar) ==="
git reset --soft HEAD~3

echo ""
echo "=== 2. Büyük dosyaları ve cache'i takipten çıkarıyoruz ==="
git rm -r --cached FacialKeypointsDetection/src/__pycache__ 2>/dev/null || true
git rm -r --cached HumanActionRecognition/src/__pycache__ 2>/dev/null || true
git rm --cached "FacialKeypointsDetection/data/raw/training.csv" 2>/dev/null || true
git rm --cached "FacialKeypointsDetection/data/raw/test.csv" 2>/dev/null || true
git rm --cached "FacialKeypointsDetection/models/final_model.pkl" 2>/dev/null || true
git rm -r --cached "FacialKeypointsDetection/models" 2>/dev/null || true
git rm -r --cached "HumanActionRecognition/models" 2>/dev/null || true
# data/raw içindeki csv'leri de çıkar (varsa)
git rm --cached "FacialKeypointsDetection/data/raw/"*.csv 2>/dev/null || true
git rm --cached "HumanActionRecognition/data/raw/"*.csv 2>/dev/null || true
# .gitkeep'leri geri ekleyeceğiz
git add FacialKeypointsDetection/data/raw/.gitkeep FacialKeypointsDetection/data/processed/.gitkeep FacialKeypointsDetection/models/.gitkeep 2>/dev/null || true
git add HumanActionRecognition/data/raw/.gitkeep HumanActionRecognition/data/processed/.gitkeep HumanActionRecognition/models/.gitkeep 2>/dev/null || true

echo ""
echo "=== 3. .gitignore ve tüm proje (ignore edilenler hariç) ==="
git add .gitignore
git add .

echo ""
echo "=== 4. Durum ==="
git status

echo ""
echo "=== 5. Tek commit atıyoruz ==="
git commit -m "Clean repo: .gitignore for data/models/cache, Kaggle data links in READMEs"

echo ""
echo "Bitti. Push için: git push origin main"
echo "Daha önce bu branch'i push ettiysen: git push origin main --force-with-lease"
