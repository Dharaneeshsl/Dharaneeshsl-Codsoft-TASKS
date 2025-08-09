// Credit Card Fraud Detection - JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    checkSystemStatus();
    
    // Form submission handler
    document.getElementById('transactionForm').addEventListener('submit', function(e) {
        e.preventDefault();
        analyzeTransaction();
    });
});

// Check system status
async function checkSystemStatus() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            document.getElementById('modelsLoaded').textContent = data.models_loaded;
            document.getElementById('systemHealth').textContent = 'Healthy';
            document.getElementById('systemHealth').className = 'status-value healthy';
        } else {
            document.getElementById('modelsLoaded').textContent = '0';
            document.getElementById('systemHealth').textContent = 'Error';
            document.getElementById('systemHealth').className = 'status-value error';
        }
    } catch (error) {
        console.error('Error checking system status:', error);
        document.getElementById('modelsLoaded').textContent = '0';
        document.getElementById('systemHealth').textContent = 'Error';
        document.getElementById('systemHealth').className = 'status-value error';
    }
}

// Analyze single transaction
async function analyzeTransaction() {
    const formData = new FormData(document.getElementById('transactionForm'));
    const transactionData = {};
    
    // Convert form data to object
    for (let [key, value] of formData.entries()) {
        if (value !== '') {
            transactionData[key] = parseFloat(value) || value;
        }
    }
    
    // Validate required fields
    if (!transactionData.amt || !transactionData.lat || !transactionData.long) {
        alert('Please fill in all required fields (Amount, Latitude, Longitude)');
        return;
    }
    
    // Show loading overlay
    showLoading();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(transactionData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayResults(result);
        } else {
            throw new Error(result.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing transaction: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Display results
function displayResults(result) {
    const resultsCard = document.getElementById('resultsCard');
    const resultsContent = document.getElementById('resultsContent');
    
    const isFraudulent = result.is_fraudulent;
    const confidence = result.confidence;
    
    // Determine confidence level
    let confidenceClass = 'confidence-low';
    if (confidence >= 0.8) {
        confidenceClass = 'confidence-high';
    } else if (confidence >= 0.6) {
        confidenceClass = 'confidence-medium';
    }
    
    // Create results HTML
    const resultsHTML = `
        <div class="result-item ${isFraudulent ? 'fraudulent' : 'legitimate'} fade-in">
            <div class="result-header">
                <div class="result-title">
                    ${isFraudulent ? '🚨 FRAUDULENT TRANSACTION' : '✅ LEGITIMATE TRANSACTION'}
                </div>
                <div class="result-confidence ${confidenceClass}">
                    ${(confidence * 100).toFixed(1)}% Confidence
                </div>
            </div>
            
            <div class="mb-20">
                <strong>Transaction Details:</strong>
                <ul style="margin-top: 10px; margin-left: 20px;">
                    <li>Amount: $${result.analysis.transaction_data.amt || 'N/A'}</li>
                    <li>Location: (${result.analysis.transaction_data.lat || 'N/A'}, ${result.analysis.transaction_data.long || 'N/A'})</li>
                    <li>Risk Level: ${result.analysis.risk_level}</li>
                </ul>
            </div>
            
            <div class="mb-20">
                <strong>Recommendations:</strong>
                <ul style="margin-top: 10px; margin-left: 20px;">
                    ${result.analysis.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
            
            <div>
                <strong>Model Predictions:</strong>
                <div style="margin-top: 10px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                    ${Object.entries(result.analysis.model_predictions).map(([model, pred]) => `
                        <div style="background: #f1f5f9; padding: 10px; border-radius: 6px; text-align: center;">
                            <div style="font-weight: 600; color: #475569;">${model.replace('_', ' ').toUpperCase()}</div>
                            <div style="color: ${pred.prediction ? '#e53e3e' : '#38a169'}; font-weight: 500;">
                                ${pred.prediction ? 'FRAUD' : 'LEGIT'}
                            </div>
                            <div style="font-size: 0.9rem; color: #64748b;">
                                ${pred.confidence ? (pred.confidence * 100).toFixed(1) + '%' : 'N/A'}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
    
    resultsContent.innerHTML = resultsHTML;
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth' });
}

// Load sample data
function loadSampleData() {
    const sampleData = {
        amt: 150.00,
        lat: 40.7128,
        long: -74.0060,
        merchant: 'online',
        hour: 14,
        day_of_week: 2
    };
    
    // Fill form with sample data
    Object.entries(sampleData).forEach(([key, value]) => {
        const element = document.getElementById(key);
        if (element) {
            element.value = value;
        }
    });
    
    // Show notification
    showNotification('Sample data loaded! Click "Analyze Transaction" to test.', 'info');
}

// Clear form
function clearForm() {
    document.getElementById('transactionForm').reset();
    document.getElementById('resultsCard').style.display = 'none';
    showNotification('Form cleared!', 'info');
}

// Analyze batch transactions
async function analyzeBatch() {
    const fileInput = document.getElementById('batchFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a CSV file');
        return;
    }
    
    if (!file.name.endsWith('.csv')) {
        alert('Please select a CSV file');
        return;
    }
    
    showLoading();
    
    try {
        const text = await file.text();
        const lines = text.split('\n');
        const headers = lines[0].split(',');
        const transactions = [];
        
        // Parse CSV data
        for (let i = 1; i < lines.length; i++) {
            if (lines[i].trim()) {
                const values = lines[i].split(',');
                const transaction = {};
                headers.forEach((header, index) => {
                    transaction[header.trim()] = values[index] ? parseFloat(values[index]) || values[index] : '';
                });
                transactions.push(transaction);
            }
        }
        
        if (transactions.length === 0) {
            throw new Error('No valid transactions found in CSV');
        }
        
        // Send batch prediction request
        const response = await fetch('/batch_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ transactions: transactions })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayBatchResults(result);
        } else {
            throw new Error(result.error || 'Batch prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing batch: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Display batch results
function displayBatchResults(result) {
    const batchResults = document.getElementById('batchResults');
    
    const resultsHTML = `
        <div class="fade-in">
            <h3>Batch Analysis Results</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0;">
                <div style="background: #f0f9ff; padding: 20px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #0369a1;">${result.total_transactions}</div>
                    <div style="color: #475569;">Total Transactions</div>
                </div>
                <div style="background: #fef2f2; padding: 20px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #dc2626;">${result.fraudulent_count}</div>
                    <div style="color: #475569;">Fraudulent</div>
                </div>
                <div style="background: #f0fdf4; padding: 20px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #16a34a;">${result.legitimate_count}</div>
                    <div style="color: #475569;">Legitimate</div>
                </div>
            </div>
            
            <div style="margin-top: 20px;">
                <strong>Fraud Rate:</strong> ${((result.fraudulent_count / result.total_transactions) * 100).toFixed(2)}%
            </div>
            
            <div style="margin-top: 20px; max-height: 300px; overflow-y: auto;">
                <strong>Individual Results:</strong>
                <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                    <thead>
                        <tr style="background: #f1f5f9;">
                            <th style="padding: 10px; text-align: left; border: 1px solid #e2e8f0;">ID</th>
                            <th style="padding: 10px; text-align: left; border: 1px solid #e2e8f0;">Prediction</th>
                            <th style="padding: 10px; text-align: left; border: 1px solid #e2e8f0;">Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${result.results.map(r => `
                            <tr>
                                <td style="padding: 10px; border: 1px solid #e2e8f0;">${r.transaction_id}</td>
                                <td style="padding: 10px; border: 1px solid #e2e8f0; color: ${r.is_fraudulent ? '#dc2626' : '#16a34a'}; font-weight: 500;">
                                    ${r.is_fraudulent ? 'FRAUD' : 'LEGIT'}
                                </td>
                                <td style="padding: 10px; border: 1px solid #e2e8f0;">${(r.confidence * 100).toFixed(1)}%</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    batchResults.innerHTML = resultsHTML;
    batchResults.style.display = 'block';
    batchResults.scrollIntoView({ behavior: 'smooth' });
}

// Show loading overlay
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

// Hide loading overlay
function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'error' ? '#fed7d7' : type === 'success' ? '#c6f6d5' : '#e2e8f0'};
        color: ${type === 'error' ? '#c53030' : type === 'success' ? '#38a169' : '#4a5568'};
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1001;
        max-width: 300px;
        animation: slideIn 0.3s ease-out;
    `;
    notification.textContent = message;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// Add slideOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style); 