from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import os
import json
import threading
import time
from pathlib import Path
import sys
import subprocess

# Don't import the step modules directly - they execute on import
# Instead, we'll use subprocess or import their functions carefully
import whisper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from llama_cpp import Llama

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ai-video-tutor-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
socketio = SocketIO(app, cors_allowed_origins="*")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MODEL_PATH = "models/Phi-3.5-mini-instruct-Q4_K_L.gguf"

# Global storage for session data
session_data = {}

# =========================
# STEP 1: Speech to Text
# =========================

def download_youtube_audio(url: str, audio_path: Path):
    """
    Downloads BEST audio from YouTube and converts to WAV.
    """
    print("⬇️ Downloading YouTube audio...")
    
    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(audio_path),
        url
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("✅ Audio downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ YouTube download failed: {e}")
        print("💡 Tip: Try using the local file upload option instead!")
        print("   Or install Node.js to fix YouTube downloads.")
        raise Exception("YouTube download failed. Please try uploading a local video file instead, or install Node.js for YouTube support.")

def transcribe_audio_file(audio_path: Path, output_path: Path):
    """Transcribe audio file using Whisper"""
    print("🧠 Loading Whisper model...")
    model = whisper.load_model("small")
    
    print("📝 Transcribing audio...")
    result = model.transcribe(str(audio_path), verbose=False)
    
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    
    print("✅ Transcription complete!")
    return segments

def process_step1(video_source, source_type, session_id):
    try:
        socketio.emit('progress', {
            'step': 1,
            'message': 'Extracting audio from video...',
            'status': 'processing'
        }, to=session_id)
        
        AUDIO_PATH = DATA_DIR / "input_audio.wav"
        OUTPUT_PATH = DATA_DIR / "transcript.json"
        
        if source_type == 'youtube':
            download_youtube_audio(video_source, AUDIO_PATH)
        else:
            # Handle local file upload
            local_path = Path(video_source)
            if local_path.exists():
                # Copy to working directory
                import shutil
                shutil.copy2(str(local_path), str(AUDIO_PATH))
            else:
                raise FileNotFoundError(f"File not found: {video_source}")
        
        socketio.emit('progress', {
            'step': 1,
            'message': 'Audio extracted! Starting transcription...',
            'status': 'processing'
        }, to=session_id)
        
        socketio.sleep(0)  # Allow socket to emit
        
        transcribe_audio_file(AUDIO_PATH, OUTPUT_PATH)
        
        socketio.emit('progress', {
            'step': 1,
            'message': 'Transcription complete!',
            'status': 'complete'
        }, to=session_id)
        
        return True
    except Exception as e:
        print(f"Error in step 1: {str(e)}")
        socketio.emit('progress', {
            'step': 1,
            'message': f'Error: {str(e)}',
            'status': 'error'
        }, to=session_id)
        return False

# =========================
# STEP 2: Topic Segmentation
# =========================

def process_step2(session_id):
    try:
        socketio.emit('progress', {
            'step': 2,
            'message': 'Analyzing content and splitting into topics...',
            'status': 'processing'
        }, to=session_id)
        
        socketio.sleep(0)
        
        TRANSCRIPT_PATH = DATA_DIR / "transcript.json"
        OUTPUT_PATH = DATA_DIR / "topics.json"
        
        with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        
        texts = [seg["text"] for seg in transcript]
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts)
        
        # Topic segmentation logic (simplified from original)
        topics = segment_topics(transcript, texts, embeddings)
        
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(topics, f, indent=2, ensure_ascii=False)
        
        socketio.emit('progress', {
            'step': 2,
            'message': f'Successfully split into {len(topics)} topics!',
            'status': 'complete',
            'data': {'topic_count': len(topics)}
        }, to=session_id)
        
        return True
    except Exception as e:
        print(f"Error in step 2: {str(e)}")
        socketio.emit('progress', {
            'step': 2,
            'message': f'Error: {str(e)}',
            'status': 'error'
        }, to=session_id)
        return False

def segment_topics(transcript, texts, embeddings):
    """Simplified topic segmentation"""
    SIMILARITY_THRESHOLD = 0.65
    MIN_TOPIC_DURATION = 45
    MAX_TOPIC_DURATION = 180
    MIN_SENTENCE_LENGTH = 20
    ROLLING_WINDOW = 3
    LOW_SIMILARITY_PATIENCE = 2
    MERGE_MIN_DURATION = 30
    MERGE_MIN_COHERENCE = 0.4
    
    def rolling_embedding(embeds, window=ROLLING_WINDOW):
        return np.mean(embeds[-window:], axis=0)
    
    def topic_duration(topic):
        return topic["end"] - topic["start"]
    
    topics = []
    current = {
        "start": transcript[0]["start"],
        "end": transcript[0]["end"],
        "texts": [texts[0]],
        "embeddings": [embeddings[0]],
        "similarities": []
    }
    
    low_similarity_count = 0
    
    for i in range(1, len(transcript)):
        text = texts[i]
        
        if len(text) < MIN_SENTENCE_LENGTH:
            current["texts"].append(text)
            current["end"] = transcript[i]["end"]
            current["embeddings"].append(embeddings[i])
            continue
        
        avg_embed = rolling_embedding(current["embeddings"])
        similarity = cosine_similarity([avg_embed], [embeddings[i]])[0][0]
        current["similarities"].append(similarity)
        
        duration = topic_duration(current)
        dynamic_threshold = max(
            0.55,
            np.mean(current["similarities"][-3:]) - 0.05
        )
        
        if similarity < dynamic_threshold:
            low_similarity_count += 1
        else:
            low_similarity_count = 0
        
        should_split = (
            (low_similarity_count >= LOW_SIMILARITY_PATIENCE and duration >= MIN_TOPIC_DURATION)
            or duration >= MAX_TOPIC_DURATION
        )
        
        if should_split:
            topics.append({
                "start": current["start"],
                "end": current["end"],
                "texts": current["texts"],
                "coherence_score": round(
                    float(np.mean(current["similarities"])) if current["similarities"] else 1.0,
                    3
                )
            })
            
            current = {
                "start": transcript[i]["start"],
                "end": transcript[i]["end"],
                "texts": [text],
                "embeddings": [embeddings[i]],
                "similarities": []
            }
            low_similarity_count = 0
        else:
            current["end"] = transcript[i]["end"]
            current["texts"].append(text)
            current["embeddings"].append(embeddings[i])
    
    topics.append({
        "start": current["start"],
        "end": current["end"],
        "texts": current["texts"],
        "coherence_score": round(
            float(np.mean(current["similarities"])) if current["similarities"] else 1.0,
            3
        )
    })
    
    # Merge small topics
    def merge_small_topics(topics):
        if not topics:
            return topics
        merged = [topics[0]]
        
        for topic in topics[1:]:
            last = merged[-1]
            duration = topic["end"] - topic["start"]
            
            if duration < MERGE_MIN_DURATION or topic["coherence_score"] < MERGE_MIN_COHERENCE:
                last["end"] = topic["end"]
                last["texts"].extend(topic["texts"])
                last["coherence_score"] = round(
                    (last["coherence_score"] + topic["coherence_score"]) / 2, 3
                )
            else:
                merged.append(topic)
        
        return merged
    
    return merge_small_topics(topics)

# =========================
# STEP 3: Topic Processing
# =========================

def process_step3(session_id):
    try:
        socketio.emit('progress', {
            'step': 3,
            'message': 'Processing topics with AI...',
            'status': 'processing'
        }, to=session_id)
        
        socketio.sleep(0)
        
        INPUT_PATH = DATA_DIR / "topics.json"
        OUTPUT_PATH = DATA_DIR / "processed_topics.json"
        
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=8,
            temperature=0.3,
            verbose=False
        )
        
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            topics = json.load(f)
        
        processed_topics = []
        
        for idx, topic in enumerate(topics):
            socketio.emit('progress', {
                'step': 3,
                'message': f'Processing topic {idx + 1} of {len(topics)}...',
                'status': 'processing',
                'data': {'current': idx + 1, 'total': len(topics)}
            }, to=session_id)
            
            socketio.sleep(0)
            
            raw_text = " ".join(topic["texts"]).strip()
            prompt = build_processing_prompt(raw_text)
            
            output = llm(prompt, max_tokens=600)
            response = output["choices"][0]["text"]
            
            try:
                start = response.find("{")
                end = response.rfind("}") + 1
                structured = json.loads(response[start:end])
            except Exception:
                structured = {
                    "clean_explanation": raw_text[:600],
                    "key_points": [],
                    "example": "",
                    "prerequisites": ["None"]
                }
            
            processed_topics.append({
                "topic_id": idx,
                "start": topic["start"],
                "end": topic["end"],
                "raw_text": raw_text,
                "clean_explanation": structured.get("clean_explanation", ""),
                "key_points": structured.get("key_points", []),
                "example": structured.get("example", ""),
                "prerequisites": structured.get("prerequisites", ["None"])
            })
        
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(processed_topics, f, indent=2, ensure_ascii=False)
        
        socketio.emit('progress', {
            'step': 3,
            'message': 'All topics processed successfully!',
            'status': 'complete'
        }, to=session_id)
        
        return True
    except Exception as e:
        print(f"Error in step 3: {str(e)}")
        socketio.emit('progress', {
            'step': 3,
            'message': f'Error: {str(e)}',
            'status': 'error'
        }, to=session_id)
        return False

def build_processing_prompt(text: str) -> str:
    return f"""
<|system|>
You are an expert teacher who explains concepts clearly to beginners.
You never introduce new concepts not present in the lecture.
<|end|>

<|user|>
Explain the following lecture segment in very simple language.

Rules:
- Do NOT add new concepts
- Be concise
- Use exactly ONE simple example
- List 3–5 key points
- Mention prerequisites or write "None"

Lecture segment:
{text}

Return STRICT JSON:
{{
  "clean_explanation": "...",
  "key_points": ["...", "..."],
  "example": "...",
  "prerequisites": ["..."]
}}
<|end|>

<|assistant|>
"""

# =========================
# STEP 5: MCQ Generation
# =========================

def process_step5(session_id):
    try:
        socketio.emit('progress', {
            'step': 5,
            'message': 'Generating quiz questions...',
            'status': 'processing'
        }, to=session_id)
        
        socketio.sleep(0)
        
        INPUT_PATH = DATA_DIR / "processed_topics.json"
        OUTPUT_PATH = DATA_DIR / "mcqs.json"
        
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=8,
            temperature=0.3,
            verbose=False
        )
        
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            topics = json.load(f)
        
        full_context = "\n".join(
            f"Topic {t['topic_id']}: {t['clean_explanation']}"
            for t in topics
        )
        
        prompt = build_mcq_prompt(full_context)
        output = llm(prompt, max_tokens=1200)
        response = output["choices"][0]["text"]
        
        start = response.find("[")
        end = response.rfind("]") + 1
        mcqs = json.loads(response[start:end])
        
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(mcqs, f, indent=2, ensure_ascii=False)
        
        socketio.emit('progress', {
            'step': 5,
            'message': f'Generated {len(mcqs)} quiz questions!',
            'status': 'complete',
            'data': {'question_count': len(mcqs)}
        }, to=session_id)
        
        return True
    except Exception as e:
        print(f"Error in step 5: {str(e)}")
        socketio.emit('progress', {
            'step': 5,
            'message': f'Error: {str(e)}',
            'status': 'error'
        }, to=session_id)
        return False

def build_mcq_prompt(context):
    return f"""
<|system|>
You are an expert educational assessment designer.

You generate multiple-choice questions based on instructional content.
You strictly follow cognitive difficulty definitions.
<|end|>

<|user|>
Instructional content:
{context}

Task:
Generate MCQs at three difficulty levels.

Difficulty definitions:
Easy:
- Recall or recognition
- Explicitly stated facts
- Low conceptual complexity

Medium:
- Application of ideas
- Connecting concepts
- Procedures demonstrated in content

Difficult:
- High abstraction
- Inference or reasoning
- Transfer to unfamiliar situations
- Multi-step reasoning not explicitly modeled

Rules:
- 3 Easy MCQs
- 3 Medium MCQs
- 3 Difficult MCQs
- Each MCQ has 4 options
- Only one correct option
- Do NOT add external knowledge

Return STRICT JSON ONLY in this format:
[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "correct_index": 0,
    "difficulty": "easy",
    "concept": "..."
  }}
]
<|end|>

<|assistant|>
"""

# =========================
# ROUTES
# =========================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lesson')
def lesson():
    return render_template('lesson.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/api/start-processing', methods=['POST'])
def start_processing():
    data = request.json
    video_source = data.get('video_source')
    source_type = data.get('source_type')
    session_id = data.get('session_id', 'default')
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_pipeline,
        args=(video_source, source_type, session_id)
    )
    thread.start()
    
    return jsonify({'status': 'started', 'session_id': session_id})

def process_pipeline(video_source, source_type, session_id):
    """Run all processing steps"""
    # Step 1: Speech to Text
    if not process_step1(video_source, source_type, session_id):
        return
    
    # Step 2: Topic Segmentation
    if not process_step2(session_id):
        return
    
    # Step 3: Topic Processing
    if not process_step3(session_id):
        return
    
    # Step 5: MCQ Generation (Step 4 is interactive, handled in lesson page)
    if not process_step5(session_id):
        return
    
    socketio.emit('progress', {
        'step': 'complete',
        'message': 'Lesson ready! Starting now...',
        'status': 'complete'
    }, to=session_id)

@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    # Save uploaded file
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    upload_folder.mkdir(exist_ok=True)
    
    filename = 'uploaded_' + str(int(time.time())) + '_' + file.filename
    filepath = upload_folder / filename
    file.save(str(filepath))
    
    return jsonify({
        'status': 'success',
        'file_path': str(filepath)
    })

@app.route('/api/transcribe-audio', methods=['POST'])
def transcribe_audio_route():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio provided'})
    
    audio_file = request.files['audio']
    
    # Save temporarily
    temp_path = DATA_DIR / 'temp_audio.wav'
    audio_file.save(str(temp_path))
    
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(str(temp_path))
        
        # Clean up
        temp_path.unlink()
        
        return jsonify({
            'status': 'success',
            'transcription': result['text']
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/get-report')
def get_report():
    try:
        with open(DATA_DIR / "final_report.json", "r", encoding="utf-8") as f:
            report = json.load(f)
        return jsonify({'status': 'success', 'report': report})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/download-lesson-notes')
def download_lesson_notes():
    """Generate and download lesson notes as PDF"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from reportlab.lib.colors import HexColor
        
        with open(DATA_DIR / "processed_topics.json", "r", encoding="utf-8") as f:
            topics = json.load(f)
        
        # Create PDF
        pdf_path = DATA_DIR / "lesson_notes.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Container for PDF elements
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#1a2332'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        topic_title_style = ParagraphStyle(
            'TopicTitle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#2563eb'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=HexColor('#10b981'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=10
        )
        
        # Title
        story.append(Paragraph("LESSON NOTES", title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Generated: {time.strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Add each topic
        for idx, topic in enumerate(topics):
            if idx > 0:
                story.append(PageBreak())
            
            # Topic title
            story.append(Paragraph(f"TOPIC {topic['topic_id'] + 1}", topic_title_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Overview
            story.append(Paragraph("Overview", heading_style))
            story.append(Paragraph(topic['clean_explanation'], body_style))
            story.append(Spacer(1, 0.15*inch))
            
            # Key points
            if topic['key_points']:
                story.append(Paragraph("Key Points", heading_style))
                for i, point in enumerate(topic['key_points'], 1):
                    story.append(Paragraph(f"{i}. {point}", body_style))
                story.append(Spacer(1, 0.15*inch))
            
            # Example
            if topic['example']:
                story.append(Paragraph("Example", heading_style))
                story.append(Paragraph(topic['example'], body_style))
                story.append(Spacer(1, 0.15*inch))
            
            # Prerequisites
            if topic['prerequisites'] and topic['prerequisites'] != ['None']:
                story.append(Paragraph("Prerequisites", heading_style))
                for prereq in topic['prerequisites']:
                    story.append(Paragraph(f"• {prereq}", body_style))
        
        # Build PDF
        doc.build(story)
        
        return send_from_directory(DATA_DIR, "lesson_notes.pdf", as_attachment=True)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/get-topics')
def get_topics():
    try:
        with open(DATA_DIR / "processed_topics.json", "r", encoding="utf-8") as f:
            topics = json.load(f)
        return jsonify({'status': 'success', 'topics': topics})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/download-performance-report')
def download_performance_report():
    """Generate and download performance report as PDF"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.colors import HexColor, white
        
        # Load report data
        with open(DATA_DIR / "final_report.json", "r", encoding="utf-8") as f:
            report_data = json.load(f)
        
        with open(DATA_DIR / "processed_topics.json", "r", encoding="utf-8") as f:
            topics = json.load(f)
        
        # Create PDF
        pdf_path = DATA_DIR / "performance_report.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#1a2332'),
            spaceAfter=10,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=HexColor('#64748b'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#2563eb'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'Body',
            parent=styles['BodyText'],
            fontSize=11,
            leading=16,
            spaceAfter=10
        )
        
        # Title
        story.append(Paragraph("STUDENT PERFORMANCE REPORT", title_style))
        story.append(Paragraph(f"Generated: {time.strftime('%B %d, %Y at %I:%M %p')}", subtitle_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Performance Summary
        story.append(Paragraph("Performance Summary", section_style))
        
        mcq_summary = report_data.get('mcq_summary', [])
        total_questions = len(mcq_summary)
        correct_answers = sum(1 for q in mcq_summary if q.get('correct', False))
        score_percent = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        
        summary_data = [
            ['Metric', 'Value'],
            ['Topics Covered', str(len(topics))],
            ['Total Questions', str(total_questions)],
            ['Correct Answers', str(correct_answers)],
            ['Quiz Score', f"{score_percent:.1f}%"]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#e2e8f0'))
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Topics Learned
        story.append(Paragraph("Topics Learned", section_style))
        for idx, topic in enumerate(topics, 1):
            story.append(Paragraph(f"<b>Topic {idx}:</b> {topic['clean_explanation'][:100]}...", body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Descriptive Evaluations
        if report_data.get('descriptive_evaluations'):
            story.append(Paragraph("Descriptive Question Feedback", section_style))
            for idx, evaluation in enumerate(report_data['descriptive_evaluations'], 1):
                story.append(Paragraph(f"<b>Question {idx}:</b>", body_style))
                story.append(Paragraph(evaluation['question'], body_style))
                story.append(Paragraph(f"<b>Your Answer:</b> {evaluation['student_answer']}", body_style))
                story.append(Paragraph(f"<b>Feedback:</b> {evaluation['feedback']}", body_style))
                story.append(Spacer(1, 0.15*inch))
        
        # Final Analysis
        story.append(Paragraph("Overall Analysis", section_style))
        final_report_text = report_data.get('final_report', 'No detailed analysis available.')
        story.append(Paragraph(final_report_text, body_style))
        
        # Build PDF
        doc.build(story)
        
        return send_from_directory(DATA_DIR, "performance_report.pdf", as_attachment=True)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/get-mcqs')
def get_mcqs():
    try:
        with open(DATA_DIR / "mcqs.json", "r", encoding="utf-8") as f:
            mcqs = json.load(f)
        return jsonify({'status': 'success', 'mcqs': mcqs})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/answer-doubt', methods=['POST'])
def answer_doubt():
    data = request.json
    question = data.get('question')
    context = data.get('context', '')
    
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=8,
            temperature=0.3,
            verbose=False
        )
        
        prompt = build_doubt_prompt(context, question)
        output = llm(prompt, max_tokens=400)
        answer = output["choices"][0]["text"].strip()
        
        return jsonify({'status': 'success', 'answer': answer})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def build_doubt_prompt(context, question):
    return f"""
<|system|>
You are a patient tutor helping a student understand a topic.
You must ONLY use the provided explanation.
You must NOT introduce new concepts.
Explain simply with one example.
<|end|>

<|user|>
Context so far:
{context}

Student question:
{question}

Answer clearly and simply.
<|end|>

<|assistant|>
"""

@app.route('/api/evaluate-mcq', methods=['POST'])
def evaluate_mcq():
    data = request.json
    question = data.get('question')
    options = data.get('options')
    chosen_index = data.get('chosen_index')
    correct_index = data.get('correct_index')
    concept = data.get('concept', '')
    
    is_correct = chosen_index == correct_index
    
    if not is_correct:
        try:
            # Load topics for explanation
            with open(DATA_DIR / "processed_topics.json", "r", encoding="utf-8") as f:
                topics = json.load(f)
            
            explanation = next(
                (t['clean_explanation'] for t in topics if str(t.get('topic_id')) == concept),
                ""
            )
            
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=2048,
                n_threads=8,
                temperature=0.3,
                verbose=False
            )
            
            prompt = build_feedback_prompt(
                question,
                options,
                options[chosen_index] if 0 <= chosen_index < len(options) else "Invalid",
                options[correct_index],
                explanation
            )
            
            output = llm(prompt, max_tokens=400)
            feedback = output["choices"][0]["text"].strip()
            
            return jsonify({
                'status': 'success',
                'is_correct': False,
                'feedback': feedback
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            })
    
    return jsonify({
        'status': 'success',
        'is_correct': True,
        'feedback': 'Correct! Well done!'
    })

def build_feedback_prompt(question, options, chosen, correct, explanation):
    return f"""
<|system|>
You are a helpful tutor giving constructive feedback.
Explain mistakes clearly without judgment.
<|end|>

<|user|>
Question:
{question}

Options:
{options}

Student chose: {chosen}
Correct answer: {correct}

Relevant explanation:
{explanation}

Explain:
1. Why the chosen option is wrong
2. Why the correct option is right
<|end|>

<|assistant|>
"""

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    data = request.json
    mcq_results = data.get('mcq_results', [])
    descriptive_answers = data.get('descriptive_answers', [])
    
    try:
        with open(DATA_DIR / "processed_topics.json", "r", encoding="utf-8") as f:
            topics = json.load(f)
        
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=8,
            temperature=0.3,
            verbose=False
        )
        
        all_explanations = "\n".join(
            f"- {t['clean_explanation']}" for t in topics
        )
        
        # Evaluate descriptive answers
        evaluations = []
        for item in descriptive_answers:
            prompt = build_evaluation_prompt(
                item['question'],
                item['answer'],
                all_explanations
            )
            output = llm(prompt, max_tokens=500)
            feedback = output["choices"][0]["text"].strip()
            
            evaluations.append({
                "question": item['question'],
                "student_answer": item['answer'],
                "feedback": feedback
            })
        
        # Generate final report
        report_prompt_text = build_report_prompt(topics, mcq_results, evaluations)
        output = llm(report_prompt_text, max_tokens=700)
        final_report = output["choices"][0]["text"].strip()
        
        report_data = {
            "mcq_summary": mcq_results,
            "descriptive_evaluations": evaluations,
            "final_report": final_report
        }
        
        with open(DATA_DIR / "final_report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({'status': 'success', 'report': report_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def build_evaluation_prompt(question, answer, content):
    return f"""
<|system|>
You are an expert teacher evaluating a student's written answer.
Be constructive and specific.
<|end|>

<|user|>
Question:
{question}

Student answer:
{answer}

Relevant instructional content:
{content}

Evaluate the answer on:
1. Conceptual correctness
2. Completeness
3. Clarity

Then give improvement suggestions.
<|end|>

<|assistant|>
"""

def build_report_prompt(topics, mcq_results, evaluations):
    topic_list = ", ".join(f"Topic {t['topic_id']}" for t in topics)
    
    return f"""
<|system|>
You are an intelligent tutoring system generating a final learning report.
<|end|>

<|user|>
Topics covered:
{topic_list}

MCQ Performance:
{json.dumps(mcq_results, indent=2)}

Descriptive evaluations:
{json.dumps(evaluations, indent=2)}

Generate a final student-facing report with:
- Topics learned
- Strengths
- Weak areas
- Final summary notes
- Study recommendations
<|end|>

<|assistant|>
"""

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('join')
def handle_join(data):
    session_id = data.get('session_id', 'default')
    from flask_socketio import join_room
    join_room(session_id)
    print(f'Client joined session: {session_id}')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)