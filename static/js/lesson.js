// Lesson.js - Interactive Lesson with Pause/Play and Auto-scroll

let topics = [];
let currentTopicIndex = 0;
let accumulatedContext = '';
let isRecording = false;
let isSpeaking = false;
let isPaused = false;
let currentUtterance = null;
let currentSentenceIndex = 0;
let sentences = [];

// DOM Elements
const menuToggle = document.getElementById('menu-toggle');
const menuOverlay = document.getElementById('menu-overlay');
const hamburgerMenu = document.getElementById('hamburger-menu');
const closeMenu = document.getElementById('close-menu');

const pausePlayBtn = document.getElementById('pause-play-btn');
const controlText = document.getElementById('control-text');

const topicNumber = document.getElementById('topic-number');
const topicTitle = document.getElementById('topic-title');
const explanationText = document.getElementById('explanation-text');
const keyPoints = document.getElementById('key-points');
const exampleBox = document.getElementById('example-box');
const exampleText = document.getElementById('example-text');
const chalkboardSection = document.getElementById('chalkboard-section');

const teacherVideo = document.getElementById('teacher-video');
const understandingCheck = document.getElementById('understanding-check');
const btnYes = document.getElementById('btn-yes');
const btnNo = document.getElementById('btn-no');

const doubtSection = document.getElementById('doubt-section');
const doubtText = document.getElementById('doubt-text');
const submitDoubt = document.getElementById('submit-doubt');
const answerBox = document.getElementById('answer-box');
const answerText = document.getElementById('answer-text');
const continueBtn = document.getElementById('continue-learning');
const answerLoading = document.getElementById('answer-loading');

const methodTabs = document.querySelectorAll('.method-tab');
const typeInput = document.querySelector('.type-input');
const speakInput = document.querySelector('.speak-input');
const voiceRecordBtn = document.getElementById('voice-record-btn');
const recordStatus = document.getElementById('record-status');

// Speech Recognition
let recognition = null;
if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-IN';
}

// Menu Toggle
menuToggle.addEventListener('click', () => {
    hamburgerMenu.classList.add('active');
    menuOverlay.classList.add('active');
});

closeMenu.addEventListener('click', () => {
    hamburgerMenu.classList.remove('active');
    menuOverlay.classList.remove('active');
});

menuOverlay.addEventListener('click', () => {
    hamburgerMenu.classList.remove('active');
    menuOverlay.classList.remove('active');
});

// Pause/Play Control
pausePlayBtn.addEventListener('click', () => {
    if (isPaused) {
        // Resume
        isPaused = false;
        pausePlayBtn.innerHTML = '<i class="fas fa-pause"></i><span id="control-text">Pause</span>';
        
        if (isSpeaking && currentUtterance) {
            window.speechSynthesis.resume();
            teacherVideo.play();
        }
    } else {
        // Pause
        isPaused = true;
        pausePlayBtn.innerHTML = '<i class="fas fa-play"></i><span id="control-text">Play</span>';
        
        if (isSpeaking) {
            window.speechSynthesis.pause();
            teacherVideo.pause();
        }
    }
});

// Input Method Tabs
methodTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        methodTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        const method = tab.dataset.method;
        if (method === 'type') {
            typeInput.classList.remove('hidden');
            speakInput.classList.add('hidden');
        } else {
            speakInput.classList.remove('hidden');
            typeInput.classList.add('hidden');
        }
    });
});

// Voice Recording
voiceRecordBtn.addEventListener('click', () => {
    if (!recognition) {
        alert('Speech recognition not supported');
        return;
    }
    
    if (!isRecording) {
        recognition.start();
        isRecording = true;
        voiceRecordBtn.classList.add('recording');
        recordStatus.textContent = 'Listening... Speak now';
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            doubtText.value = transcript;
        };
        
        recognition.onerror = () => {
            isRecording = false;
            voiceRecordBtn.classList.remove('recording');
            recordStatus.textContent = 'Click to start recording';
        };
        
        recognition.onend = () => {
            isRecording = false;
            voiceRecordBtn.classList.remove('recording');
            recordStatus.textContent = 'Click to start recording';
        };
    } else {
        recognition.stop();
    }
});

// Split text into sentences for synchronized scrolling
function splitIntoSentences(text) {
    return text.match(/[^\.!\?]+[\.!\?]+/g) || [text];
}

// Auto-scroll as voice reads
function scrollToElement(element) {
    if (element) {
        element.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center',
            inline: 'nearest'
        });
    }
}

// Text-to-Speech with Auto-scroll
function speakText(text, elements, onComplete) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        const voices = window.speechSynthesis.getVoices();
        
        const indianVoice = voices.find(v => 
            v.lang.includes('en-IN') && v.name.toLowerCase().includes('female')
        ) || voices.find(v => v.lang.includes('en-IN'));
        
        if (indianVoice) utterance.voice = indianVoice;
        
        utterance.rate = 0.9;
        utterance.pitch = 1.1;
        utterance.volume = 1;
        
        // Track current element being read
        let currentElementIndex = 0;
        const sentences = splitIntoSentences(text);
        const wordsPerElement = Math.ceil(text.split(' ').length / elements.length);
        
        utterance.onboundary = (event) => {
            if (event.name === 'word' && elements.length > 0) {
                const wordIndex = Math.floor(event.charIndex / (text.length / sentences.length));
                const elementIndex = Math.min(
                    Math.floor(wordIndex / (sentences.length / elements.length)),
                    elements.length - 1
                );
                
                if (elementIndex !== currentElementIndex && elementIndex >= 0) {
                    currentElementIndex = elementIndex;
                    scrollToElement(elements[elementIndex]);
                }
            }
        };
        
        utterance.onstart = () => {
            isSpeaking = true;
            isPaused = false;
            teacherVideo.loop = true;
            teacherVideo.play();
            
            // Scroll to first element
            if (elements.length > 0) {
                scrollToElement(elements[0]);
            }
        };
        
        utterance.onend = () => {
            isSpeaking = false;
            teacherVideo.loop = false;
            teacherVideo.pause();
            
            if (onComplete) {
                onComplete();
            }
        };
        
        currentUtterance = utterance;
        window.speechSynthesis.speak(utterance);
    } else if (onComplete) {
        onComplete();
    }
}

window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();

// Load Topics
async function loadTopics() {
    try {
        const response = await fetch('/api/get-topics');
        const data = await response.json();
        
        if (data.status === 'success') {
            topics = data.topics;
            displayTopic(0);
        } else {
            alert('Error loading lesson content');
        }
    } catch (error) {
        alert('Error loading lesson content');
    }
}

// Display Topic
function displayTopic(index) {
    if (index >= topics.length) {
        window.location.href = '/quiz';
        return;
    }
    
    currentTopicIndex = index;
    const topic = topics[index];
    
    topicNumber.textContent = `Topic ${index + 1} of ${topics.length}`;
    topicTitle.textContent = `Topic ${index + 1}`;
    
    // Build content to speak (without "Topic X")
    let contentToSpeak = '';
    let elementsToScroll = [];
    
    // Explanation
    if (topic.clean_explanation) {
        const explainPara = document.createElement('p');
        explainPara.style.fontSize = '1.25rem';
        explainPara.style.lineHeight = '2';
        explainPara.textContent = topic.clean_explanation;
        explanationText.innerHTML = '';
        explanationText.appendChild(explainPara);
        
        contentToSpeak += topic.clean_explanation + '. ';
        elementsToScroll.push(explainPara);
    }
    
    accumulatedContext += '\n\n' + topic.clean_explanation;
    
    // Key points
    keyPoints.innerHTML = '';
    if (topic.key_points && topic.key_points.length > 0) {
        topic.key_points.forEach((point, idx) => {
            const li = document.createElement('li');
            li.textContent = point;
            keyPoints.appendChild(li);
            contentToSpeak += `${point}. `;
            elementsToScroll.push(li);
        });
    }
    
    // Example
    if (topic.example && topic.example.trim() !== '') {
        exampleBox.style.display = 'block';
        exampleText.textContent = topic.example;
        contentToSpeak += `For example, ${topic.example}`;
        elementsToScroll.push(exampleBox);
    } else {
        exampleBox.style.display = 'none';
    }
    
    // Reset UI
    understandingCheck.classList.add('hidden');
    doubtSection.classList.add('hidden');
    answerBox.classList.add('hidden');
    doubtText.value = '';
    isPaused = false;
    pausePlayBtn.innerHTML = '<i class="fas fa-pause"></i><span id="control-text">Pause</span>';
    
    // Scroll to top
    chalkboardSection.scrollTop = 0;
    
    // Start speaking after delay
    setTimeout(() => {
        speakText(contentToSpeak, elementsToScroll, () => {
            setTimeout(() => {
                understandingCheck.classList.remove('hidden');
                understandingCheck.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 500);
        });
    }, 1000);
}

btnYes.addEventListener('click', () => {
    window.speechSynthesis.cancel();
    teacherVideo.pause();
    displayTopic(currentTopicIndex + 1);
});

btnNo.addEventListener('click', () => {
    window.speechSynthesis.cancel();
    teacherVideo.pause();
    understandingCheck.classList.add('hidden');
    doubtSection.classList.remove('hidden');
    setTimeout(() => {
        doubtSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 300);
});

submitDoubt.addEventListener('click', async () => {
    const question = doubtText.value.trim();
    
    if (!question) {
        alert('Please enter your question');
        return;
    }
    
    answerLoading.classList.remove('hidden');
    
    try {
        const response = await fetch('/api/answer-doubt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, context: accumulatedContext })
        });
        
        const data = await response.json();
        answerLoading.classList.add('hidden');
        
        if (data.status === 'success') {
            const answerPara = document.createElement('p');
            answerPara.style.lineHeight = '1.8';
            answerPara.style.fontSize = '1.125rem';
            answerPara.textContent = data.answer;
            answerText.innerHTML = '';
            answerText.appendChild(answerPara);
            answerBox.classList.remove('hidden');
            
            setTimeout(() => {
                answerBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 300);
            
            // Speak answer with scroll
            speakText(data.answer, [answerPara]);
        } else {
            alert('Error: ' + data.message);
        }
    } catch (error) {
        answerLoading.classList.add('hidden');
        alert('Error submitting question');
    }
});

continueBtn.addEventListener('click', () => {
    window.speechSynthesis.cancel();
    teacherVideo.pause();
    displayTopic(currentTopicIndex + 1);
});

window.addEventListener('beforeunload', () => {
    window.speechSynthesis.cancel();
    if (teacherVideo) teacherVideo.pause();
});

loadTopics();