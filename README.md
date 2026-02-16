# AI Video Tutor - Interactive EdTech Platform

A sophisticated web-based intelligent tutoring system that transforms any video into an interactive learning experience with AI-powered explanations, adaptive quizzing, and personalized feedback.

## Features

### 🎓 Core Functionality
- **Video/Audio Processing**: Support for YouTube links and local video/audio files
- **Automatic Transcription**: Speech-to-text conversion using Whisper AI
- **Topic Segmentation**: Intelligent content splitting using semantic similarity
- **AI-Powered Explanations**: Clear, beginner-friendly explanations using Phi-3.5-mini
- **Interactive Learning**: Doubt resolution with voice/text input
- **Adaptive Assessment**: Progressive difficulty quiz system (Easy → Medium → Hard)
- **Personalized Feedback**: Real-time evaluation and remediation
- **Comprehensive Reports**: Detailed performance analysis with strengths and improvement areas

### 🎨 Design Features
- **Classroom Theme**: Clean, minimalistic, professional interface
- **Chalkboard Interface**: Visual lesson presentation on virtual chalkboard
- **Teacher Video Integration**: Synchronized video playback
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Hamburger Navigation**: Easy access to all sections
- **Real-time Progress**: Live updates during processing
- **Voice Input/Output**: Speech recognition and text-to-speech

## Architecture

### Frontend
- **HTML5/CSS3**: Semantic markup with modern CSS features
- **Vanilla JavaScript**: No framework dependencies for better performance
- **Socket.IO**: Real-time bidirectional communication
- **Web Speech API**: Voice input and text-to-speech
- **Responsive Grid**: CSS Grid and Flexbox for layout

### Backend
- **Flask**: Python web framework
- **Flask-SocketIO**: WebSocket support for real-time updates
- **Whisper**: OpenAI's speech recognition model
- **Sentence Transformers**: Semantic text embeddings
- **llama.cpp**: Efficient LLM inference with Phi-3.5-mini
- **scikit-learn**: Cosine similarity for topic segmentation

## Installation

### Prerequisites
- Python 3.9 or higher
- FFmpeg (for audio processing)
- At least 8GB RAM
- 4+ CPU cores recommended

### Step 1: Clone and Setup

```bash
# Create project directory
mkdir ai-video-tutor
cd ai-video-tutor

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

### Step 3: Download AI Model

Create a `models` directory and download the Phi-3.5-mini model:

```bash
mkdir models
cd models

# Download Phi-3.5-mini-instruct Q4_K_L GGUF
# Get from: https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf
# Or use wget:
wget https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-Q4_K_L.gguf
```

### Step 4: Place Your Files

Copy your existing Python step files to the project root:
- step_1_speech_to_text.py
- step_2_topic_segmentation.py
- step_3_topic_processing.py
- step_4_interactive_tutor.py
- step_5_mcq_generation.py
- step_6_adaptive_evaluation.py
- step_7_final_evaluation_and_report.py

Place the teacher video:
```bash
# Copy teacher_video.mp4 to static folder
cp /path/to/teacher_video.mp4 static/
```

### Step 5: Create Required Directories

```bash
mkdir data
mkdir uploads
```

## Usage

### Starting the Server

```bash
# Activate virtual environment if not already active
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the Flask app
python app.py
```

The server will start on `http://localhost:5000`

### Using the Application

1. **Welcome Screen**
   - Choose between YouTube link or local file upload
   - Paste YouTube URL or drag & drop video file
   - Click "Start Learning"

2. **Processing Phase**
   - Watch progress as the system:
     - Extracts audio from video
     - Transcribes speech to text
     - Analyzes and segments content into topics
     - Generates AI-powered explanations
     - Creates adaptive quiz questions

3. **Learning Phase**
   - View topic content on the virtual chalkboard
   - Watch synchronized teacher video
   - Read key points and examples
   - Click "Yes" to continue or "No" to ask questions
   - Use voice or text input for doubts
   - Receive AI-powered answers with text-to-speech

4. **Assessment Phase**
   - Answer 3 easy questions
   - Progress to medium difficulty
   - Complete hard questions
   - Receive instant feedback on each answer
   - Get remediation if needed
   - Answer descriptive questions

5. **Report Phase**
   - View comprehensive performance summary
   - See topics learned and quiz scores
   - Review strengths and areas for improvement
   - Read detailed analysis and recommendations
   - Download report as PDF
   - Start a new lesson

## File Structure

```
ai-video-tutor/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── step_1_speech_to_text.py       # Original processing scripts
├── step_2_topic_segmentation.py
├── step_3_topic_processing.py
├── step_4_interactive_tutor.py
├── step_5_mcq_generation.py
├── step_6_adaptive_evaluation.py
├── step_7_final_evaluation_and_report.py
│
├── templates/                      # HTML templates
│   ├── index.html                 # Welcome page
│   ├── lesson.html                # Interactive lesson page
│   ├── quiz.html                  # Adaptive quiz page
│   └── report.html                # Final report page
│
├── static/                         # Static assets
│   ├── css/
│   │   └── style.css              # Main stylesheet
│   ├── js/
│   │   ├── main.js                # Welcome page logic
│   │   ├── lesson.js              # Lesson page logic
│   │   ├── quiz.js                # Quiz page logic
│   │   └── report.js              # Report page logic
│   └── teacher_video.mp4          # Teacher video (place here)
│
├── data/                           # Generated data (auto-created)
│   ├── input_audio.wav
│   ├── transcript.json
│   ├── topics.json
│   ├── processed_topics.json
│   ├── mcqs.json
│   └── final_report.json
│
├── models/                         # AI models
│   └── Phi-3.5-mini-instruct-Q4_K_L.gguf
│
└── uploads/                        # User uploaded files
```

## Customization

### Adjusting Processing Parameters

Edit the constants in `app.py` or individual step files:

```python
# Topic segmentation thresholds
SIMILARITY_THRESHOLD = 0.65
MIN_TOPIC_DURATION = 45
MAX_TOPIC_DURATION = 180

# Quiz pass threshold
PASS_THRESHOLD = 0.67  # 67% to pass each level

# Model parameters
n_ctx = 2048          # Context window
n_threads = 8         # CPU threads
temperature = 0.3     # Generation randomness
```

### Changing Theme Colors

Edit `static/css/style.css`:

```css
:root {
    --primary-dark: #1a2332;
    --primary-blue: #2563eb;
    --accent-green: #10b981;
    --chalkboard-green: #2d5a3d;
    /* ... modify as needed */
}
```

### Adding New Features

The modular architecture makes it easy to extend:

1. **New Routes**: Add to `app.py`
2. **New Pages**: Create template in `templates/`
3. **New Styles**: Extend `static/css/style.css`
4. **New Logic**: Add JavaScript to `static/js/`

## API Endpoints

### POST /api/start-processing
Start video processing pipeline
```json
{
    "video_source": "URL or file path",
    "source_type": "youtube or upload",
    "session_id": "unique_session_id"
}
```

### GET /api/get-topics
Retrieve processed topics

### GET /api/get-mcqs
Retrieve generated quiz questions

### POST /api/answer-doubt
Submit student question for AI response
```json
{
    "question": "Student's question",
    "context": "Accumulated learning context"
}
```

### POST /api/evaluate-mcq
Evaluate student's answer
```json
{
    "question": "Question text",
    "options": ["A", "B", "C", "D"],
    "chosen_index": 0,
    "correct_index": 1,
    "concept": "concept_id"
}
```

### POST /api/generate-report
Generate final learning report
```json
{
    "mcq_results": [...],
    "descriptive_answers": [...]
}
```

## Troubleshooting

### Issue: FFmpeg not found
**Solution**: Install FFmpeg and ensure it's in your system PATH

### Issue: Model loading fails
**Solution**: Verify model file exists in `models/` directory and path is correct

### Issue: Slow processing
**Solution**: 
- Increase `n_threads` in model initialization
- Use smaller video files for testing
- Consider using GPU-enabled versions of models

### Issue: Socket.IO connection fails
**Solution**: Check firewall settings and ensure port 5000 is accessible

### Issue: Voice input not working
**Solution**: 
- Grant microphone permissions in browser
- Use HTTPS in production (HTTP works for localhost)
- Check browser compatibility

## Performance Tips

1. **First Run**: Initial model loading takes time. Subsequent runs are faster.
2. **Video Length**: Longer videos take more time to process. Start with 5-10 minute videos.
3. **CPU Usage**: Adjust `n_threads` based on your CPU cores.
4. **Memory**: Monitor RAM usage. Close other applications if needed.

## Browser Compatibility

- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support (macOS/iOS)
- Opera: ✅ Full support

## Security Notes

For production deployment:
1. Enable HTTPS
2. Add authentication
3. Implement rate limiting
4. Validate all user inputs
5. Set up CORS properly
6. Use environment variables for secrets

## License

This project is created for educational purposes.

## Support

For issues or questions:
1. Check this README thoroughly
2. Review console logs for errors
3. Verify all dependencies are installed
4. Ensure model files are present

## Credits

- **Whisper**: OpenAI
- **Phi-3.5**: Microsoft
- **Sentence Transformers**: Hugging Face
- **Icons**: Font Awesome