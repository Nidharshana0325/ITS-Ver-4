// Report.js - Final Report Functionality

// DOM Elements
const menuToggle = document.getElementById('menu-toggle');
const menuOverlay = document.getElementById('menu-overlay');
const hamburgerMenu = document.getElementById('hamburger-menu');
const closeMenu = document.getElementById('close-menu');

const totalTopics = document.getElementById('total-topics');
const quizScore = document.getElementById('quiz-score');
const questionsAnswered = document.getElementById('questions-answered');
const timeSpent = document.getElementById('time-spent');

const topicsLearned = document.getElementById('topics-learned');
const strengthsList = document.getElementById('strengths-list');
const improvementsList = document.getElementById('improvements-list');
const detailedAnalysis = document.getElementById('detailed-analysis');
const recommendations = document.getElementById('recommendations');

const descriptiveFeedbackSection = document.getElementById('descriptive-feedback-section');
const descriptiveFeedback = document.getElementById('descriptive-feedback');

const downloadReportBtn = document.getElementById('download-report');
const downloadNotesBtn = document.getElementById('download-notes');
const startNewLessonBtn = document.getElementById('start-new-lesson');

const reportLoading = document.getElementById('report-loading');

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

// Download Lesson Notes
downloadNotesBtn.addEventListener('click', () => {
    window.location.href = '/api/download-lesson-notes';
});

// Download Report
downloadReportBtn.addEventListener('click', () => {
    window.location.href = '/api/download-performance-report';
});

// Load Report Data
async function loadReport() {
    // Try to get from session storage first
    const storedData = sessionStorage.getItem('reportData');
    
    if (storedData) {
        const reportData = JSON.parse(storedData);
        displayReport(reportData);
    } else {
        // If no stored data, try to fetch from server
        try {
            const response = await fetch('/api/get-report');
            const data = await response.json();
            
            if (data.status === 'success') {
                displayReport(data.report);
            } else {
                console.error('No report data available');
                showEmptyReport();
            }
        } catch (error) {
            console.error('Error loading report:', error);
            showEmptyReport();
        }
    }
}

// Display Report
function displayReport(reportData) {
    // Performance Summary
    if (reportData.mcq_summary) {
        const totalQuestions = reportData.mcq_summary.length;
        const correctAnswers = reportData.mcq_summary.filter(q => q.correct).length;
        const scorePercent = totalQuestions > 0 ? Math.round((correctAnswers / totalQuestions) * 100) : 0;
        
        questionsAnswered.textContent = totalQuestions;
        quizScore.textContent = `${scorePercent}%`;
    }
    
    // Load topics
    loadTopicsData();
    
    // Descriptive Evaluations
    if (reportData.descriptive_evaluations && reportData.descriptive_evaluations.length > 0) {
        descriptiveFeedbackSection.style.display = 'block';
        
        descriptiveFeedback.innerHTML = '';
        reportData.descriptive_evaluations.forEach((evaluation, index) => {
            const div = document.createElement('div');
            div.style.marginBottom = 'var(--spacing-xl)';
            div.style.padding = 'var(--spacing-lg)';
            div.style.background = 'var(--paper-cream)';
            div.style.borderRadius = '12px';
            div.innerHTML = `
                <h4 style="color: var(--primary-blue); margin-bottom: var(--spacing-md);">
                    Question ${index + 1}
                </h4>
                <p style="font-style: italic; color: var(--text-muted); margin-bottom: var(--spacing-md);">
                    "${evaluation.question}"
                </p>
                <p style="margin-bottom: var(--spacing-md);">
                    <strong>Your Answer:</strong><br>
                    ${evaluation.student_answer}
                </p>
                <div style="background: white; padding: var(--spacing-md); border-radius: 8px; border-left: 4px solid var(--primary-blue);">
                    <strong style="color: var(--primary-blue);">Feedback:</strong><br>
                    <p style="margin-top: var(--spacing-xs); line-height: 1.8;">
                        ${evaluation.feedback}
                    </p>
                </div>
            `;
            descriptiveFeedback.appendChild(div);
        });
    }
    
    // Final Report Analysis
    if (reportData.final_report) {
        parseAndDisplayAnalysis(reportData.final_report);
    } else {
        generateDefaultAnalysis(reportData);
    }
}

// Load Topics Data
async function loadTopicsData() {
    try {
        const response = await fetch('/api/get-topics');
        const data = await response.json();
        
        if (data.status === 'success') {
            const topics = data.topics;
            totalTopics.textContent = topics.length;
            
            // Display topics learned
            topicsLearned.innerHTML = '';
            topics.forEach((topic, index) => {
                const li = document.createElement('li');
                li.className = 'topic-learned-item';
                li.innerHTML = `
                    <strong>Topic ${index + 1}:</strong> 
                    ${topic.clean_explanation.substring(0, 100)}...
                `;
                topicsLearned.appendChild(li);
            });
        }
    } catch (error) {
        console.error('Error loading topics:', error);
    }
}

// Parse and Display Analysis
function parseAndDisplayAnalysis(reportText) {
    // This is a simplified parser - in production, you'd want more robust parsing
    detailedAnalysis.innerHTML = `<p style="line-height: 1.8;">${reportText}</p>`;
    
    // Extract strengths and improvements (basic extraction)
    const strengthsPattern = /strength[s]?:?\s*(.*?)(?=weak|improve|recommendation|$)/is;
    const improvementsPattern = /weak|improve[ment]*:?\s*(.*?)(?=recommendation|strength|$)/is;
    const recommendationsPattern = /recommendation[s]?:?\s*(.*?)$/is;
    
    const strengthsMatch = reportText.match(strengthsPattern);
    const improvementsMatch = reportText.match(improvementsPattern);
    const recommendationsMatch = reportText.match(recommendationsPattern);
    
    if (strengthsMatch && strengthsMatch[1]) {
        displayList(strengthsList, strengthsMatch[1]);
    } else {
        displayDefaultStrengths();
    }
    
    if (improvementsMatch && improvementsMatch[1]) {
        displayList(improvementsList, improvementsMatch[1]);
    } else {
        displayDefaultImprovements();
    }
    
    if (recommendationsMatch && recommendationsMatch[1]) {
        recommendations.innerHTML = `<p style="line-height: 1.8;">${recommendationsMatch[1].trim()}</p>`;
    } else {
        displayDefaultRecommendations();
    }
}

// Display List Helper
function displayList(element, text) {
    const items = text.split(/[•\-\n]/).filter(item => item.trim().length > 10);
    
    element.innerHTML = '';
    items.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item.trim();
        element.appendChild(li);
    });
}

// Generate Default Analysis
function generateDefaultAnalysis(reportData) {
    if (reportData.mcq_summary) {
        const totalQuestions = reportData.mcq_summary.length;
        const correctAnswers = reportData.mcq_summary.filter(q => q.correct).length;
        const scorePercent = totalQuestions > 0 ? (correctAnswers / totalQuestions) * 100 : 0;
        
        detailedAnalysis.innerHTML = `
            <p style="line-height: 1.8;">
                You have completed the interactive learning session successfully. 
                You answered ${correctAnswers} out of ${totalQuestions} questions correctly, 
                achieving a score of ${Math.round(scorePercent)}%. 
                ${scorePercent >= 80 ? 'Excellent work!' : scorePercent >= 60 ? 'Good effort!' : 'Keep practicing to improve your understanding.'}
            </p>
        `;
    }
    
    displayDefaultStrengths();
    displayDefaultImprovements();
    displayDefaultRecommendations();
}

// Default Displays
function displayDefaultStrengths() {
    strengthsList.innerHTML = `
        <li>Completed all lesson topics</li>
        <li>Actively participated in the learning process</li>
        <li>Attempted all quiz questions</li>
    `;
}

function displayDefaultImprovements() {
    improvementsList.innerHTML = `
        <li>Review difficult concepts for better understanding</li>
        <li>Practice more questions on challenging topics</li>
        <li>Take time to understand feedback provided</li>
    `;
}

function displayDefaultRecommendations() {
    recommendations.innerHTML = `
        <p style="line-height: 1.8;">
            To strengthen your understanding:
        </p>
        <ul style="list-style: disc; margin-left: var(--spacing-lg); margin-top: var(--spacing-md);">
            <li style="margin-bottom: var(--spacing-xs);">Revisit the lesson content and take notes</li>
            <li style="margin-bottom: var(--spacing-xs);">Practice similar problems to reinforce concepts</li>
            <li style="margin-bottom: var(--spacing-xs);">Explore additional resources on challenging topics</li>
            <li style="margin-bottom: var(--spacing-xs);">Try teaching the concepts to someone else</li>
        </ul>
    `;
}

// Show Empty Report
function showEmptyReport() {
    detailedAnalysis.innerHTML = `
        <p style="text-align: center; color: var(--text-muted); padding: var(--spacing-xl);">
            <i class="fas fa-exclamation-circle" style="font-size: 3rem; margin-bottom: var(--spacing-md);"></i><br>
            No report data available. Please complete a lesson first.
        </p>
    `;
}

// Start New Lesson
startNewLessonBtn.addEventListener('click', () => {
    // Clear session storage
    sessionStorage.clear();
    
    // Redirect to home
    window.location.href = '/';
});

// Initialize
loadReport();