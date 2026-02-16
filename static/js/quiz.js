// Quiz.js - Adaptive Quiz with Retry Logic and Full Explanations

let mcqs = [];
let currentDifficulty = 'easy';
let currentQuestionIndex = 0;
let currentLevelQuestions = [];
let levelResults = {
    easy: { correct: 0, total: 0, attempts: 0 },
    medium: { correct: 0, total: 0, attempts: 0 },
    difficult: { correct: 0, total: 0, attempts: 0 }
};
let selectedAnswer = null;
let hasAnswered = false;
let allResults = [];
let descriptiveAnswers = [];

const PASS_THRESHOLD = 0.67; // Must get 2 out of 3 correct

// DOM Elements
const menuToggle = document.getElementById('menu-toggle');
const menuOverlay = document.getElementById('menu-overlay');
const hamburgerMenu = document.getElementById('hamburger-menu');
const closeMenu = document.getElementById('close-menu');

const difficultyBadge = document.getElementById('difficulty-badge');
const quizProgress = document.getElementById('quiz-progress');
const questionNumber = document.getElementById('question-number');
const questionText = document.getElementById('question-text');
const optionsList = document.getElementById('options-list');
const feedbackBox = document.getElementById('feedback-box');
const submitAnswerBtn = document.getElementById('submit-answer');

const questionCard = document.getElementById('question-card');
const levelCompleteCard = document.getElementById('level-complete-card');
const levelResultTitle = document.getElementById('level-result-title');
const levelResultMessage = document.getElementById('level-result-message');
const levelScore = document.getElementById('level-score');
const levelCorrect = document.getElementById('level-correct');
const levelTotal = document.getElementById('level-total');
const remediationSection = document.getElementById('remediation-section');
const remediationContent = document.getElementById('remediation-content');
const continueQuizBtn = document.getElementById('continue-quiz');
const retryLevelBtn = document.getElementById('retry-level');

const descriptiveCard = document.getElementById('descriptive-card');
const descriptiveQuestions = document.getElementById('descriptive-questions');
const submitDescriptiveBtn = document.getElementById('submit-descriptive');

const evaluationLoading = document.getElementById('evaluation-loading');
const loadingTitle = document.getElementById('loading-title');
const loadingMessage = document.getElementById('loading-message');

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

// Load MCQs
async function loadMCQs() {
    try {
        const response = await fetch('/api/get-mcqs');
        const data = await response.json();
        
        if (data.status === 'success') {
            mcqs = data.mcqs;
            startLevel('easy');
        } else {
            alert('Error loading quiz questions');
        }
    } catch (error) {
        alert('Error loading quiz questions');
    }
}

// Start Level
function startLevel(difficulty) {
    currentDifficulty = difficulty;
    currentLevelQuestions = mcqs.filter(q => q.difficulty === difficulty);
    currentQuestionIndex = 0;
    levelResults[difficulty].attempts++;
    levelResults[difficulty].correct = 0;
    levelResults[difficulty].total = 0;
    
    difficultyBadge.textContent = difficulty.charAt(0).toUpperCase() + difficulty.slice(1);
    difficultyBadge.className = 'difficulty-badge ' + difficulty;
    
    questionCard.classList.remove('hidden');
    levelCompleteCard.classList.add('hidden');
    descriptiveCard.classList.add('hidden');
    
    displayQuestion();
}

// Display Question
function displayQuestion() {
    if (currentQuestionIndex >= currentLevelQuestions.length) {
        showLevelComplete();
        return;
    }
    
    const question = currentLevelQuestions[currentQuestionIndex];
    
    selectedAnswer = null;
    hasAnswered = false;
    submitAnswerBtn.disabled = true;
    feedbackBox.classList.add('hidden');
    document.getElementById('feedback-container').classList.add('hidden');
    
    quizProgress.textContent = `Question ${currentQuestionIndex + 1} of ${currentLevelQuestions.length}`;
    questionNumber.textContent = `Question ${currentQuestionIndex + 1}`;
    questionText.textContent = question.question;
    
    // Scroll to top of question
    questionCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    optionsList.innerHTML = '';
    question.options.forEach((option, index) => {
        const li = document.createElement('li');
        li.className = 'option-item';
        
        const btn = document.createElement('button');
        btn.className = 'option-btn';
        btn.dataset.index = index;
        
        const label = document.createElement('span');
        label.className = 'option-label';
        label.textContent = String.fromCharCode(65 + index);
        
        const text = document.createElement('span');
        text.textContent = option;
        
        btn.appendChild(label);
        btn.appendChild(text);
        btn.addEventListener('click', () => selectOption(index));
        
        li.appendChild(btn);
        optionsList.appendChild(li);
    });
}

// Select Option
function selectOption(index) {
    if (hasAnswered) return;
    
    selectedAnswer = index;
    submitAnswerBtn.disabled = false;
    
    const optionBtns = optionsList.querySelectorAll('.option-btn');
    optionBtns.forEach((btn, i) => {
        if (i === index) {
            btn.classList.add('selected');
        } else {
            btn.classList.remove('selected');
        }
    });
}

// Submit Answer
submitAnswerBtn.addEventListener('click', async () => {
    if (selectedAnswer === null || hasAnswered) return;
    
    hasAnswered = true;
    const question = currentLevelQuestions[currentQuestionIndex];
    const isCorrect = selectedAnswer === question.correct_index;
    
    if (isCorrect) {
        levelResults[currentDifficulty].correct++;
    }
    levelResults[currentDifficulty].total++;
    
    allResults.push({
        question: question.question,
        difficulty: currentDifficulty,
        correct: isCorrect,
        concept: question.concept
    });
    
    evaluationLoading.classList.remove('hidden');
    loadingTitle.textContent = 'Evaluating Your Answer';
    loadingMessage.textContent = isCorrect ? 'Great job!' : 'Let me explain...';
    
    try {
        const response = await fetch('/api/evaluate-mcq', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question.question,
                options: question.options,
                chosen_index: selectedAnswer,
                correct_index: question.correct_index,
                concept: question.concept || currentDifficulty
            })
        });
        
        const data = await response.json();
        evaluationLoading.classList.add('hidden');
        
        if (data.status === 'success') {
            const optionBtns = optionsList.querySelectorAll('.option-btn');
            optionBtns.forEach((btn, i) => {
                btn.disabled = true;
                if (i === question.correct_index) {
                    btn.classList.add('correct');
                } else if (i === selectedAnswer && !isCorrect) {
                    btn.classList.add('incorrect');
                }
            });
            
            if (isCorrect) {
                // Show simple success feedback
                feedbackBox.classList.remove('hidden');
                feedbackBox.className = 'feedback-box correct';
                feedbackBox.innerHTML = `
                    <h4>✓ Correct!</h4>
                    <p style="margin-top: var(--spacing-sm); line-height: 1.8;">${data.feedback}</p>
                `;
                
                // Auto-scroll to feedback
                setTimeout(() => {
                    feedbackBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }, 300);
            } else {
                // Show two-box layout for wrong answer
                const feedbackContainer = document.getElementById('feedback-container');
                const wrongExplanation = document.getElementById('wrong-explanation');
                const correctExplanation = document.getElementById('correct-explanation');
                
                // Parse feedback to separate wrong and correct explanations
                const feedbackText = data.feedback;
                
                // Try to split feedback intelligently
                let wrongText = '';
                let correctText = '';
                
                if (feedbackText.includes('correct answer') || feedbackText.includes('right answer')) {
                    const parts = feedbackText.split(/correct answer|right answer/i);
                    wrongText = parts[0].trim();
                    correctText = 'The correct answer' + (parts[1] || '').trim();
                } else {
                    // Fallback: split by sentences
                    const sentences = feedbackText.match(/[^.!?]+[.!?]+/g) || [feedbackText];
                    const mid = Math.ceil(sentences.length / 2);
                    wrongText = sentences.slice(0, mid).join(' ');
                    correctText = sentences.slice(mid).join(' ');
                }
                
                wrongExplanation.textContent = wrongText || 'Your chosen answer does not match the correct concept from the lesson.';
                correctExplanation.textContent = correctText || feedbackText;
                
                feedbackContainer.classList.remove('hidden');
                
                // Auto-scroll to feedback
                setTimeout(() => {
                    feedbackContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 300);
            }
            
            submitAnswerBtn.textContent = 'Next Question';
            submitAnswerBtn.disabled = false;
            submitAnswerBtn.onclick = () => {
                currentQuestionIndex++;
                displayQuestion();
                submitAnswerBtn.textContent = 'Submit Answer';
                submitAnswerBtn.onclick = null;
            };
        }
    } catch (error) {
        evaluationLoading.classList.add('hidden');
        alert('Error evaluating answer');
    }
});

// Show Level Complete
function showLevelComplete() {
    questionCard.classList.add('hidden');
    levelCompleteCard.classList.remove('hidden');
    
    const result = levelResults[currentDifficulty];
    const score = result.total > 0 ? (result.correct / result.total) : 0;
    const passed = score >= PASS_THRESHOLD;
    
    levelScore.textContent = `${Math.round(score * 100)}%`;
    levelCorrect.textContent = result.correct;
    levelTotal.textContent = result.total;
    
    if (passed) {
        levelResultTitle.textContent = '🎉 Level Complete!';
        levelResultMessage.textContent = `Great job! You've passed the ${currentDifficulty} level.`;
        
        retryLevelBtn.classList.add('hidden');
        continueQuizBtn.classList.remove('hidden');
        remediationSection.classList.add('hidden');
        
        continueQuizBtn.onclick = () => {
            if (currentDifficulty === 'easy') {
                startLevel('medium');
            } else if (currentDifficulty === 'medium') {
                startLevel('difficult');
            } else {
                showDescriptiveQuestions();
            }
        };
    } else {
        levelResultTitle.textContent = 'Keep Trying!';
        levelResultMessage.textContent = `You need at least ${Math.ceil(PASS_THRESHOLD * result.total)} correct to pass. Let's review and try again.`;
        
        continueQuizBtn.classList.add('hidden');
        retryLevelBtn.classList.remove('hidden');
        remediationSection.classList.remove('hidden');
        
        showRemediation();
        
        retryLevelBtn.onclick = () => {
            startLevel(currentDifficulty);
        };
    }
}

// Show Remediation
async function showRemediation() {
    const weakConcepts = allResults
        .filter(r => !r.correct && r.difficulty === currentDifficulty)
        .map(r => r.concept)
        .filter((v, i, a) => a.indexOf(v) === i);
    
    if (weakConcepts.length === 0) {
        remediationContent.innerHTML = '<p>Review the lesson content again.</p>';
        return;
    }
    
    remediationContent.innerHTML = `
        <p style="margin-bottom: var(--spacing-md);">
            Please review these concepts:
        </p>
        <ul style="list-style: disc; margin-left: var(--spacing-lg);">
            ${weakConcepts.map(c => `<li style="margin-bottom: var(--spacing-xs);">Topic ${c}</li>`).join('')}
        </ul>
        <p style="margin-top: var(--spacing-md); font-style: italic; color: var(--text-muted);">
            Take your time to understand before retrying.
        </p>
    `;
}

// Show Descriptive Questions
function showDescriptiveQuestions() {
    questionCard.classList.add('hidden');
    levelCompleteCard.classList.add('hidden');
    descriptiveCard.classList.remove('hidden');
    
    const questions = [
        "Explain the most important concept you learned from this lesson in your own words.",
        "Choose one concept from the lesson and explain why it is important, using an example."
    ];
    
    descriptiveQuestions.innerHTML = '';
    questions.forEach((q, index) => {
        const div = document.createElement('div');
        div.style.marginBottom = 'var(--spacing-xl)';
        div.innerHTML = `
            <h3 style="color: var(--primary-dark); margin-bottom: var(--spacing-md);">
                Question ${index + 1}
            </h3>
            <p style="font-size: 1.125rem; margin-bottom: var(--spacing-md); color: var(--text-muted);">
                ${q}
            </p>
            <textarea 
                class="doubt-textarea descriptive-answer" 
                placeholder="Write your answer here..."
                data-question="${q}"
                style="min-height: 150px;"
            ></textarea>
        `;
        descriptiveQuestions.appendChild(div);
    });
}

// Submit Descriptive Answers
submitDescriptiveBtn.addEventListener('click', async () => {
    const answerElements = document.querySelectorAll('.descriptive-answer');
    descriptiveAnswers = [];
    
    let allAnswered = true;
    answerElements.forEach(el => {
        const answer = el.value.trim();
        if (!answer) {
            allAnswered = false;
        } else {
            descriptiveAnswers.push({
                question: el.dataset.question,
                answer: answer
            });
        }
    });
    
    if (!allAnswered) {
        alert('Please answer all questions');
        return;
    }
    
    evaluationLoading.classList.remove('hidden');
    loadingTitle.textContent = 'Evaluating Your Answers';
    loadingMessage.textContent = 'Analyzing your understanding...';
    
    try {
        const response = await fetch('/api/generate-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mcq_results: allResults,
                descriptive_answers: descriptiveAnswers
            })
        });
        
        const data = await response.json();
        evaluationLoading.classList.add('hidden');
        
        if (data.status === 'success') {
            sessionStorage.setItem('reportData', JSON.stringify(data.report));
            window.location.href = '/report';
        } else {
            alert('Error: ' + data.message);
        }
    } catch (error) {
        evaluationLoading.classList.add('hidden');
        alert('Error generating report');
    }
});

loadMCQs();