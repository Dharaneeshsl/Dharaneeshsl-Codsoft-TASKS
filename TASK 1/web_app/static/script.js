// Movie Genre Classification - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const predictionForm = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const predictAllBtn = document.getElementById('predictAllBtn');
    const moviePlotTextarea = document.getElementById('moviePlot');
    const modelSelect = document.getElementById('modelSelect');
    const resultsCard = document.getElementById('resultsCard');
    const comparisonCard = document.getElementById('comparisonCard');
    const loadingOverlay = document.getElementById('loadingOverlay');

    // Event listeners
    predictionForm.addEventListener('submit', handlePrediction);
    predictAllBtn.addEventListener('click', handlePredictAll);

    // Handle single model prediction
    async function handlePrediction(e) {
        e.preventDefault();
        
        const text = moviePlotTextarea.value.trim();
        const model = modelSelect.value;
        
        if (!text) {
            showAlert('Please enter a movie plot summary.', 'error');
            return;
        }
        
        showLoading(true);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    model: model
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                displayResults(data);
                hideComparison();
            } else {
                showAlert(data.error || 'Prediction failed.', 'error');
            }
        } catch (error) {
            console.error('Error:', error);
            showAlert('An error occurred while making the prediction.', 'error');
        } finally {
            showLoading(false);
        }
    }

    // Handle prediction with all models
    async function handlePredictAll() {
        const text = moviePlotTextarea.value.trim();
        
        if (!text) {
            showAlert('Please enter a movie plot summary.', 'error');
            return;
        }
        
        showLoading(true);
        
        try {
            const response = await fetch('/predict_all', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                displayComparison(data.predictions);
                hideResults();
            } else {
                showAlert(data.error || 'Comparison failed.', 'error');
            }
        } catch (error) {
            console.error('Error:', error);
            showAlert('An error occurred while comparing models.', 'error');
        } finally {
            showLoading(false);
        }
    }

    // Display prediction results
    function displayResults(data) {
        const { prediction, top_predictions, features } = data;
        
        // Display main prediction
        const mainPredictionDiv = document.getElementById('mainPrediction');
        mainPredictionDiv.innerHTML = `
            <div>${prediction.genre}</div>
            <div class="confidence">Confidence: ${(prediction.confidence * 100).toFixed(1)}%</div>
            <div style="font-size: 0.9rem; margin-top: 5px;">Model: ${prediction.model.replace('_', ' ').toUpperCase()}</div>
        `;
        
        // Display top predictions
        const topPredictionsDiv = document.getElementById('topPredictions');
        topPredictionsDiv.innerHTML = top_predictions.map(pred => `
            <div class="prediction-item">
                <span class="genre">${pred.genre}</span>
                <span class="confidence">${(pred.confidence * 100).toFixed(1)}%</span>
            </div>
        `).join('');
        
        // Display feature analysis
        const featuresDiv = document.getElementById('featuresList');
        if (features && features.length > 0) {
            featuresDiv.innerHTML = features.map(feature => `
                <div class="feature-item">
                    <span class="feature">${feature.feature}</span>
                    <span class="importance">${feature.importance.toFixed(4)}</span>
                </div>
            `).join('');
        } else {
            featuresDiv.innerHTML = '<p style="color: #666; font-style: italic;">Feature analysis not available for this model.</p>';
        }
        
        // Show results card
        resultsCard.style.display = 'block';
        resultsCard.scrollIntoView({ behavior: 'smooth' });
    }

    // Display model comparison
    function displayComparison(predictions) {
        const comparisonTableDiv = document.getElementById('comparisonTable');
        
        const tableHTML = `
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Predicted Genre</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    ${Object.entries(predictions).map(([modelName, pred]) => {
                        const confidenceClass = getConfidenceClass(pred.confidence);
                        return `
                            <tr>
                                <td class="model-name">${formatModelName(modelName)}</td>
                                <td>${pred.genre}</td>
                                <td class="${confidenceClass}">${(pred.confidence * 100).toFixed(1)}%</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        `;
        
        comparisonTableDiv.innerHTML = tableHTML;
        
        // Show comparison card
        comparisonCard.style.display = 'block';
        comparisonCard.scrollIntoView({ behavior: 'smooth' });
    }

    // Helper functions
    function formatModelName(modelName) {
        return modelName
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    function getConfidenceClass(confidence) {
        if (confidence >= 0.7) return 'confidence-high';
        if (confidence >= 0.5) return 'confidence-medium';
        return 'confidence-low';
    }

    function hideResults() {
        resultsCard.style.display = 'none';
    }

    function hideComparison() {
        comparisonCard.style.display = 'none';
    }

    function showLoading(show) {
        loadingOverlay.style.display = show ? 'flex' : 'none';
    }

    function showAlert(message, type = 'info') {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.innerHTML = `
            <i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;
        
        // Insert at the top of main content
        const mainContent = document.querySelector('.main-content');
        mainContent.insertBefore(alertDiv, mainContent.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    // Add some interactive features
    moviePlotTextarea.addEventListener('input', function() {
        // Auto-resize textarea
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 300) + 'px';
    });

    // Add example text on double-click
    moviePlotTextarea.addEventListener('dblclick', function() {
        if (!this.value.trim()) {
            this.value = 'A young wizard discovers his magical heritage and must defeat an evil sorcerer to save the world. Along the way, he learns about friendship, courage, and the power of love.';
            this.dispatchEvent(new Event('input'));
        }
    });

    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl+Enter to submit
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            predictionForm.dispatchEvent(new Event('submit'));
        }
        
        // Ctrl+Shift+Enter to compare all models
        if (e.ctrlKey && e.shiftKey && e.key === 'Enter') {
            e.preventDefault();
            handlePredictAll();
        }
    });

    // Add tooltips for buttons
    predictBtn.title = 'Predict genre using selected model (Ctrl+Enter)';
    predictAllBtn.title = 'Compare predictions from all models (Ctrl+Shift+Enter)';

    // Initialize with some helpful text
    console.log('Movie Genre Classification Web App loaded successfully!');
    console.log('Keyboard shortcuts:');
    console.log('- Ctrl+Enter: Submit prediction');
    console.log('- Ctrl+Shift+Enter: Compare all models');
    console.log('- Double-click textarea: Load example text');
}); 