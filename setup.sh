#!/bin/bash

# AI Video Tutor Setup Script
# This script helps set up the entire project

echo "========================================="
echo "   AI Video Tutor - Setup Script"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.9 or higher.${NC}"
    exit 1
fi

# Check FFmpeg
echo ""
echo -e "${BLUE}Checking FFmpeg...${NC}"
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}FFmpeg is not installed.${NC}"
    echo "Please install FFmpeg:"
    echo "  macOS: brew install ffmpeg"
    echo "  Linux: sudo apt install ffmpeg"
    echo "  Windows: choco install ffmpeg"
    exit 1
else
    echo -e "${GREEN}FFmpeg is installed${NC}"
fi

# Create virtual environment
echo ""
echo -e "${BLUE}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo ""
echo -e "${BLUE}Installing Python dependencies...${NC}"
echo "This may take several minutes..."
pip install --upgrade pip
pip install -r requirements.txt

# Create required directories
echo ""
echo -e "${BLUE}Creating required directories...${NC}"
mkdir -p data
mkdir -p uploads
mkdir -p models
mkdir -p static/css
mkdir -p static/js
mkdir -p templates

echo -e "${GREEN}Directories created${NC}"

# Check for model file
echo ""
echo -e "${BLUE}Checking for AI model...${NC}"
if [ ! -f "models/Phi-3.5-mini-instruct-Q4_K_L.gguf" ]; then
    echo -e "${RED}Model file not found!${NC}"
    echo ""
    echo "Please download the Phi-3.5-mini-instruct model:"
    echo "  1. Visit: https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf"
    echo "  2. Download: Phi-3.5-mini-instruct-Q4_K_L.gguf"
    echo "  3. Place it in the 'models/' directory"
    echo ""
    echo "Or use wget:"
    echo "  wget -P models/ https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-Q4_K_L.gguf"
else
    echo -e "${GREEN}Model file found${NC}"
fi

# Check for step files
echo ""
echo -e "${BLUE}Checking for processing step files...${NC}"
missing_files=0
for i in {1..7}; do
    if [ ! -f "step_${i}_*.py" ]; then
        echo -e "${RED}Missing: step_${i}_*.py${NC}"
        missing_files=1
    fi
done

if [ $missing_files -eq 1 ]; then
    echo ""
    echo "Please ensure all step files (step_1 through step_7) are in the project root."
fi

# Check for teacher video
echo ""
echo -e "${BLUE}Checking for teacher video...${NC}"
if [ ! -f "static/teacher_video.mp4" ]; then
    echo -e "${RED}Teacher video not found!${NC}"
    echo "Please place 'teacher_video.mp4' in the 'static/' directory"
else
    echo -e "${GREEN}Teacher video found${NC}"
fi

# Summary
echo ""
echo "========================================="
echo "   Setup Summary"
echo "========================================="
echo ""
echo -e "${GREEN}✓ Python 3 installed${NC}"
echo -e "${GREEN}✓ FFmpeg installed${NC}"
echo -e "${GREEN}✓ Virtual environment created${NC}"
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo -e "${GREEN}✓ Directories created${NC}"

if [ ! -f "models/Phi-3.5-mini-instruct-Q4_K_L.gguf" ]; then
    echo -e "${RED}✗ AI model missing${NC}"
else
    echo -e "${GREEN}✓ AI model present${NC}"
fi

if [ ! -f "static/teacher_video.mp4" ]; then
    echo -e "${RED}✗ Teacher video missing${NC}"
else
    echo -e "${GREEN}✓ Teacher video present${NC}"
fi

echo ""
echo "========================================="
echo ""
echo "To start the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run the server: python app.py"
echo "  3. Open browser: http://localhost:5000"
echo ""
echo "========================================="