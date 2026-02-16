// Main.js - Welcome Page Functionality

const socket = io();
let sessionId = 'session_' + Date.now();
let selectedFile = null;
let currentSourceType = 'youtube';

// DOM Elements
const youtubTab = document.querySelector('[data-tab="youtube"]');
const uploadTab = document.querySelector('[data-tab="upload"]');
const youtubeContent = document.querySelector('.youtube-content');
const uploadContent = document.querySelector('.upload-content');
const youtubeInput = document.getElementById('youtube-url');
const fileInput = document.getElementById('file-input');
const fileUploadArea = document.getElementById('file-upload-area');
const fileNameDisplay = document.getElementById('file-name');
const startBtn = document.getElementById('start-btn');
const progressScreen = document.getElementById('progress-screen');

// Tab Switching
youtubTab.addEventListener('click', () => {
    youtubTab.classList.add('active');
    uploadTab.classList.remove('active');
    youtubeContent.classList.remove('hidden');
    uploadContent.classList.add('hidden');
    currentSourceType = 'youtube';
});

uploadTab.addEventListener('click', () => {
    uploadTab.classList.add('active');
    youtubTab.classList.remove('active');
    uploadContent.classList.remove('hidden');
    youtubeContent.classList.add('hidden');
    currentSourceType = 'upload';
});

// File Upload
fileUploadArea.addEventListener('click', () => {
    fileInput.click();
});

fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadArea.style.borderColor = 'var(--primary-blue)';
    fileUploadArea.style.background = 'rgba(37, 99, 235, 0.05)';
});

fileUploadArea.addEventListener('dragleave', () => {
    fileUploadArea.style.borderColor = 'var(--border-light)';
    fileUploadArea.style.background = 'white';
});

fileUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadArea.style.borderColor = 'var(--border-light)';
    fileUploadArea.style.background = 'white';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        selectedFile = files[0];
        fileNameDisplay.textContent = `Selected: ${selectedFile.name}`;
        fileNameDisplay.style.marginTop = 'var(--spacing-md)';
        fileNameDisplay.style.color = 'var(--accent-green)';
        fileNameDisplay.style.fontWeight = '500';
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        selectedFile = e.target.files[0];
        fileNameDisplay.textContent = `Selected: ${selectedFile.name}`;
        fileNameDisplay.style.marginTop = 'var(--spacing-md)';
        fileNameDisplay.style.color = 'var(--accent-green)';
        fileNameDisplay.style.fontWeight = '500';
    }
});

// Socket.IO - Join session
socket.emit('join', { session_id: sessionId });

// Progress tracking
let processStartTime = null;
let stepStartTimes = {};
let stepEstimates = {
    1: 45,      // 45 seconds average
    2: 120,     // 2 minutes
    3: 45,      // 45 seconds
    4: 210,     // 3.5 minutes (longest step)
    5: 90       // 1.5 minutes
};
let totalEstimate = Object.values(stepEstimates).reduce((a, b) => a + b, 0); // ~8.5 minutes
let completedSteps = [];

function updateProgressBar() {
    const progressBarFill = document.getElementById('progress-bar-fill');
    const progressPercentage = document.getElementById('progress-percentage');
    const progressTime = document.getElementById('progress-time');
    
    // Calculate progress based on completed steps
    let progress = 0;
    let elapsedTime = 0;
    
    completedSteps.forEach(step => {
        progress += (stepEstimates[step] / totalEstimate) * 100;
        elapsedTime += stepEstimates[step];
    });
    
    // Add partial progress for current active step
    const activeStep = document.querySelector('.progress-step.active');
    if (activeStep && processStartTime) {
        const stepId = parseInt(activeStep.id.split('-')[1]);
        if (stepStartTimes[stepId]) {
            const stepElapsed = (Date.now() - stepStartTimes[stepId]) / 1000;
            const stepProgress = Math.min(stepElapsed / stepEstimates[stepId], 0.95); // Cap at 95%
            progress += (stepProgress * stepEstimates[stepId] / totalEstimate) * 100;
        }
    }
    
    // Update progress bar
    progress = Math.min(Math.round(progress), 99); // Cap at 99% until complete
    progressBarFill.style.width = progress + '%';
    progressPercentage.textContent = progress + '%';
    
    // Calculate time remaining
    if (processStartTime) {
        const elapsed = (Date.now() - processStartTime) / 1000;
        const remaining = Math.max(totalEstimate - elapsed, 10);
        const minutes = Math.floor(remaining / 60);
        const seconds = Math.floor(remaining % 60);
        
        if (minutes > 0) {
            progressTime.textContent = `~${minutes}m ${seconds}s remaining`;
        } else {
            progressTime.textContent = `~${seconds}s remaining`;
        }
    }
}

// Update progress bar every second
let progressInterval = null;

// Socket.IO - Progress updates
socket.on('progress', (data) => {
    console.log('Progress update:', data);
    
    const progressMessage = document.getElementById('progress-message');
    progressMessage.textContent = data.message;
    
    // Start timer on first progress update
    if (!processStartTime && data.step >= 1) {
        processStartTime = Date.now();
        progressInterval = setInterval(updateProgressBar, 1000);
    }
    
    // Update step status
    if (data.step >= 1 && data.step <= 5) {
        const stepElement = document.getElementById(`step-${data.step}`);
        const stepTimeElement = document.getElementById(`step-${data.step}-time`);
        
        if (stepElement) {
            if (data.status === 'complete') {
                stepElement.classList.add('completed');
                stepElement.classList.remove('active');
                if (!completedSteps.includes(data.step)) {
                    completedSteps.push(data.step);
                }
                
                // Update step time with actual duration
                if (stepStartTimes[data.step]) {
                    const duration = (Date.now() - stepStartTimes[data.step]) / 1000;
                    const minutes = Math.floor(duration / 60);
                    const seconds = Math.floor(duration % 60);
                    if (minutes > 0) {
                        stepTimeElement.textContent = `✓ Completed in ${minutes}m ${seconds}s`;
                    } else {
                        stepTimeElement.textContent = `✓ Completed in ${seconds}s`;
                    }
                }
            } else if (data.status === 'processing') {
                stepElement.classList.add('active');
                stepElement.classList.remove('completed');
                if (!stepStartTimes[data.step]) {
                    stepStartTimes[data.step] = Date.now();
                }
            }
        }
        
        updateProgressBar();
    }
    
    // When all steps complete, redirect to lesson
    if (data.step === 'complete' && data.status === 'complete') {
        clearInterval(progressInterval);
        
        // Set progress to 100%
        document.getElementById('progress-bar-fill').style.width = '100%';
        document.getElementById('progress-percentage').textContent = '100%';
        document.getElementById('progress-time').textContent = 'Complete!';
        
        setTimeout(() => {
            window.location.href = '/lesson';
        }, 2000);
    }
});

// Start Learning Button
startBtn.addEventListener('click', async () => {
    let videoSource = '';
    
    if (currentSourceType === 'youtube') {
        videoSource = youtubeInput.value.trim();
        if (!videoSource) {
            alert('Please enter a YouTube URL');
            return;
        }
    } else {
        if (!selectedFile) {
            alert('Please select a video file');
            return;
        }
        
        // Upload file to server
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        try {
            const uploadResponse = await fetch('/api/upload-file', {
                method: 'POST',
                body: formData
            });
            
            const uploadData = await uploadResponse.json();
            if (uploadData.status === 'success') {
                videoSource = uploadData.file_path;
            } else {
                alert('File upload failed');
                return;
            }
        } catch (error) {
            console.error('Upload error:', error);
            alert('Error uploading file');
            return;
        }
    }
    
    // Show progress screen
    document.querySelector('.welcome-screen').style.display = 'none';
    progressScreen.classList.remove('hidden');
    
    // Start processing
    try {
        const response = await fetch('/api/start-processing', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                video_source: videoSource,
                source_type: currentSourceType,
                session_id: sessionId
            })
        });
        
        const data = await response.json();
        console.log('Processing started:', data);
    } catch (error) {
        console.error('Error starting processing:', error);
        alert('Error starting processing');
        progressScreen.classList.add('hidden');
        document.querySelector('.welcome-screen').style.display = 'flex';
    }
});

// Store session ID for next pages
sessionStorage.setItem('sessionId', sessionId);