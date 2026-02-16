# AI Video Tutor - Quick Start Guide

## What You're Getting

A complete web-based AI tutoring system that transforms videos into interactive lessons with:
- Automatic transcription
- Topic segmentation
- AI explanations
- Interactive Q&A
- Adaptive quizzes
- Performance reports

## Immediate Next Steps

### 1. Copy Your Existing Files

Place these files in the project root directory:
```
step_1_speech_to_text.py
step_2_topic_segmentation.py
step_3_topic_processing.py
step_4_interactive_tutor.py
step_5_mcq_generation.py
step_6_adaptive_evaluation.py
step_7_final_evaluation_and_report.py
```

### 2. Add Teacher Video

Place your `teacher_video.mp4` file in the `static/` folder

### 3. Download AI Model

Download Phi-3.5-mini-instruct model (~2.4GB):
```bash
wget -P models/ https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-Q4_K_L.gguf
```

Or manually:
1. Visit: https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf
2. Download: Phi-3.5-mini-instruct-Q4_K_L.gguf
3. Place in `models/` directory

### 4. Install Dependencies

Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Start the Server

```bash
source venv/bin/activate  # If not already active
python app.py
```

Visit: http://localhost:5000

## Project Structure Overview

```
ai-video-tutor/
├── app.py                    # Main Flask server (NEW)
├── requirements.txt          # Python dependencies (NEW)
├── setup.sh                  # Setup script (NEW)
├── README.md                 # Full documentation (NEW)
│
├── templates/                # HTML templates (NEW)
│   ├── index.html           # Welcome page
│   ├── lesson.html          # Interactive lesson
│   ├── quiz.html            # Adaptive quiz
│   └── report.html          # Final report
│
├── static/                   # Frontend assets (NEW)
│   ├── css/style.css        # Styling
│   ├── js/
│   │   ├── main.js
│   │   ├── lesson.js
│   │   ├── quiz.js
│   │   └── report.js
│   └── teacher_video.mp4    # (YOU NEED TO ADD THIS)
│
├── step_*.py                # Your existing files (COPY THESE)
├── models/                  # AI models (DOWNLOAD MODEL HERE)
├── data/                    # Auto-generated data
└── uploads/                 # User uploads
```

## Key Features Implemented

### 1. Welcome Screen
- Clean, professional classroom theme
- YouTube link or file upload options
- Real-time progress tracking

### 2. Interactive Lesson
- Virtual chalkboard with topic content
- Synchronized teacher video
- Voice/text input for questions
- AI-powered answers with speech output
- Smooth topic progression

### 3. Adaptive Quiz
- Progressive difficulty (Easy → Medium → Hard)
- Instant feedback on each question
- Automatic remediation for wrong answers
- Must pass each level to progress
- Descriptive questions at the end

### 4. Final Report
- Performance summary with stats
- Topics learned
- Strengths and weaknesses
- Detailed analysis
- Study recommendations
- Downloadable report

### 5. Navigation
- Hamburger menu for easy navigation
- Responsive design for all devices
- Clean, minimalistic interface
- Professional classroom aesthetics

## How It Works

### Backend Processing Flow:
1. **User uploads video/YouTube link**
2. **Step 1**: Extract audio and transcribe
3. **Step 2**: Segment into topics using AI
4. **Step 3**: Generate clean explanations
5. **Step 4**: Enable interactive Q&A
6. **Step 5**: Generate adaptive MCQs
7. **Step 6**: Run evaluation with feedback
8. **Step 7**: Create final report

### Frontend User Flow:
1. **Welcome** → Choose input source
2. **Processing** → Watch real-time progress
3. **Lesson** → Learn topic by topic with AI tutor
4. **Quiz** → Complete adaptive assessment
5. **Report** → View performance and recommendations

## Important Notes

### File Integration
The new web interface integrates your existing step files WITHOUT modifying them. It imports and uses them as modules.

### Teacher Video
The video should be in MP4 format. If your video is large:
- Consider compressing it
- Or use a shorter demo video for testing
- The video plays in sync with lesson content

### Model Requirements
- Phi-3.5-mini-instruct (~2.4GB)
- Runs on CPU (no GPU required)
- First load takes time, subsequent runs faster
- Adjust `n_threads` based on your CPU

### Memory Requirements
- Minimum: 8GB RAM
- Recommended: 16GB RAM
- More memory = better performance

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### FFmpeg not found
Install FFmpeg for your OS (see README.md)

### Model file errors
Verify file exists and path matches in code

### Port already in use
Change port in app.py:
```python
socketio.run(app, debug=True, host='0.0.0.0', port=5001)
```

## Testing

Start with a short video (5-10 minutes) for initial testing.

Example YouTube videos to try:
- Educational content (math, science, history)
- Tutorial videos
- Lecture recordings

## Customization

### Change Colors
Edit `static/css/style.css` root variables

### Adjust Quiz Difficulty
Modify `PASS_THRESHOLD` in app.py or quiz.js

### Update Topic Segmentation
Adjust thresholds in step files or app.py

## What's Next?

Once running:
1. Test with a sample video
2. Check all features work
3. Customize styling if desired
4. Add authentication for production
5. Deploy to a server

## Support

Check:
1. README.md for detailed documentation
2. Console logs for error messages
3. Browser developer tools for frontend issues

## Credits

Built using:
- Flask (Python web framework)
- Whisper AI (transcription)
- Phi-3.5 (AI explanations)
- Sentence Transformers (topic segmentation)
- Modern HTML/CSS/JS (frontend)

---

**Ready to start? Run `./setup.sh` and follow the prompts!**