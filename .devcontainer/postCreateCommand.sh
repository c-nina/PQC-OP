#!/bin/bash

sudo apt-get update
sudo apt-get install -y software-properties-common
echo "deb http://deb.debian.org/debian testing main" | sudo tee /etc/apt/sources.list.d/testing.list
sudo apt-get update
sudo apt-get install -y -t testing libstdc++6

# 2. Instalar dependencias de Python
pip3 install --upgrade pip
pip3 install -e ".[dev]"

# 3. Otros comandos
npm install -g @openai/codex
npm install -g @anthropic-ai/claude-code
