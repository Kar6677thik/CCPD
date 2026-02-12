/**
 * Fraud Detection Agent - Frontend Application
 */

const API_BASE = '';

// Charts
let comparisonChart = null;
let distributionChart = null;

// State
let modelsReady = false;

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    loadDataStats();
    loadModelStatus();
    checkTrainingStatus();
});

function initCharts() {
    // Model Comparison Chart
    const compCtx = document.getElementById('comparisonChart').getContext('2d');
    comparisonChart = new Chart(compCtx, {
        type: 'bar',
        data: {
            labels: ['Precision', 'Recall', 'F1 Score', 'AUC'],
            datasets: [
                {
                    label: 'Isolation Forest',
                    data: [0, 0, 0, 0],
                    backgroundColor: 'rgba(99, 102, 241, 0.7)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Autoencoder',
                    data: [0, 0, 0, 0],
                    backgroundColor: 'rgba(139, 92, 246, 0.7)',
                    borderColor: 'rgba(139, 92, 246, 1)',
                    borderWidth: 1
                },
                {
                    label: 'XGBoost',
                    data: [0, 0, 0, 0],
                    backgroundColor: 'rgba(16, 185, 129, 0.7)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#a0a0b0',
                        padding: 15,
                        font: { size: 11 }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#a0a0b0' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: { color: '#a0a0b0' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });

    // Distribution Chart
    const distCtx = document.getElementById('distributionChart').getContext('2d');
    distributionChart = new Chart(distCtx, {
        type: 'doughnut',
        data: {
            labels: ['Normal', 'Fraud'],
            datasets: [{
                data: [0, 0],
                backgroundColor: [
                    'rgba(99, 102, 241, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgba(99, 102, 241, 1)',
                    'rgba(239, 68, 68, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#a0a0b0',
                        padding: 15,
                        font: { size: 11 }
                    }
                }
            }
        }
    });
}

// ============================================
// Data Loading
// ============================================

async function loadDataStats() {
    try {
        const response = await fetch(`${API_BASE}/api/data/distribution`);
        const data = await response.json();

        document.getElementById('ds-total').textContent = formatNumber(data.total_transactions);
        document.getElementById('ds-fraud').textContent = formatNumber(data.distribution.fraud.count);
        document.getElementById('ds-normal').textContent = formatNumber(data.distribution.normal.count);
        document.getElementById('ds-ratio').textContent = data.distribution.fraud.percentage.toFixed(2) + '%';

        // Update chart
        distributionChart.data.datasets[0].data = [
            data.distribution.normal.count,
            data.distribution.fraud.count
        ];
        distributionChart.update();
    } catch (error) {
        console.error('Failed to load data stats:', error);
    }
}

async function loadModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/train/models`);
        const models = await response.json();

        // Update status badges
        updateModelStatus('status-if', models.isolation_forest?.is_trained);
        updateModelStatus('status-ae', models.autoencoder?.is_trained);
        updateModelStatus('status-xgb', models.xgboost?.is_trained);

        modelsReady = Object.values(models).every(m => m?.is_trained);

        if (modelsReady) {
            loadMetrics();
            loadFeatureImportance();
        }
    } catch (error) {
        console.error('Failed to load model status:', error);
    }
}

function updateModelStatus(elementId, isTrained) {
    const el = document.getElementById(elementId);
    if (isTrained) {
        el.textContent = 'Trained ✓';
        el.classList.add('trained');
    } else {
        el.textContent = 'Not Trained';
        el.classList.remove('trained');
    }
}

async function loadMetrics() {
    try {
        const response = await fetch(`${API_BASE}/api/metrics/`);
        const data = await response.json();

        if (data.latest) {
            const m = data.latest;
            document.getElementById('stat-accuracy').textContent = (m.accuracy * 100).toFixed(1) + '%';
            document.getElementById('stat-precision').textContent = (m.precision * 100).toFixed(1) + '%';
            document.getElementById('stat-recall').textContent = (m.recall * 100).toFixed(1) + '%';
            document.getElementById('stat-auc').textContent = m.auc_roc ? m.auc_roc.toFixed(3) : '--';
        }
    } catch (error) {
        console.error('Failed to load metrics:', error);
    }
}

async function loadFeatureImportance() {
    try {
        const response = await fetch(`${API_BASE}/api/metrics/feature-importance`);
        const data = await response.json();

        if (data.top_features) {
            const container = document.getElementById('feature-bars');
            container.innerHTML = '';

            const features = Object.entries(data.top_features).slice(0, 8);
            const maxVal = Math.max(...features.map(([_, v]) => v));

            features.forEach(([name, value]) => {
                const percent = (value / maxVal) * 100;
                container.innerHTML += `
                    <div class="feature-bar">
                        <span class="feature-name">${name}</span>
                        <div class="feature-fill-container">
                            <div class="feature-fill" style="width: ${percent}%"></div>
                        </div>
                        <span class="feature-value">${(value * 100).toFixed(1)}%</span>
                    </div>
                `;
            });
        }
    } catch (error) {
        console.error('Failed to load feature importance:', error);
    }
}

// ============================================
// Training
// ============================================

async function trainModels() {
    const modal = document.getElementById('training-modal');
    const progressBar = document.getElementById('training-progress-bar');
    const statusText = document.getElementById('training-status');
    const log = document.getElementById('training-log');

    modal.classList.remove('hidden');
    log.innerHTML = '<p>Starting training...</p>';
    progressBar.style.width = '5%';

    try {
        const response = await fetch(`${API_BASE}/api/train/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ apply_smote: true, test_size: 0.2 })
        });

        const data = await response.json();
        log.innerHTML += `<p>Training initiated: ${JSON.stringify(data.config)}</p>`;

        // Poll for status
        pollTrainingStatus(progressBar, statusText, log);
    } catch (error) {
        log.innerHTML += `<p class="error">Error: ${error.message}</p>`;
        statusText.textContent = 'Training failed';
    }
}

async function pollTrainingStatus(progressBar, statusText, log) {
    const modal = document.getElementById('training-modal');

    const poll = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/train/status`);
            const data = await response.json();

            progressBar.style.width = `${data.progress}%`;
            statusText.textContent = data.status;

            if (data.is_training) {
                log.innerHTML += `<p>Progress: ${data.progress}% - ${data.status}</p>`;
                log.scrollTop = log.scrollHeight;
                setTimeout(poll, 1000);
            } else {
                if (data.last_result) {
                    log.innerHTML += `<p class="success">Training completed!</p>`;
                    log.innerHTML += `<p>Samples: ${data.last_result.train_samples} train, ${data.last_result.test_samples} test</p>`;
                    if (data.last_result.evaluation_metrics) {
                        const m = data.last_result.evaluation_metrics;
                        log.innerHTML += `<p>Accuracy: ${(m.accuracy * 100).toFixed(2)}%</p>`;
                        log.innerHTML += `<p>Precision: ${(m.precision * 100).toFixed(2)}%</p>`;
                        log.innerHTML += `<p>Recall: ${(m.recall * 100).toFixed(2)}%</p>`;
                    }
                }

                // Refresh data
                loadModelStatus();
                loadMetrics();
                loadFeatureImportance();
                refreshComparison();

                setTimeout(() => modal.classList.add('hidden'), 3000);
                showToast('Models trained successfully!', 'success');
            }
        } catch (error) {
            log.innerHTML += `<p class="error">Error checking status: ${error.message}</p>`;
        }
    };

    poll();
}

async function checkTrainingStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/train/status`);
        const data = await response.json();

        if (data.is_training) {
            // Training in progress, show modal
            const modal = document.getElementById('training-modal');
            modal.classList.remove('hidden');
            pollTrainingStatus(
                document.getElementById('training-progress-bar'),
                document.getElementById('training-status'),
                document.getElementById('training-log')
            );
        }
    } catch (error) {
        console.error('Failed to check training status:', error);
    }
}

async function loadModels() {
    try {
        await fetch(`${API_BASE}/api/train/load`, { method: 'POST' });
        loadModelStatus();
        showToast('Models loaded from disk', 'success');
    } catch (error) {
        showToast('Failed to load models', 'error');
    }
}

// ============================================
// Detection
// ============================================

async function detectFraud() {
    const transaction = {
        Amount: parseFloat(document.getElementById('input-amount').value) || 0,
        V1: parseFloat(document.getElementById('input-v1').value) || 0,
        V2: parseFloat(document.getElementById('input-v2').value) || 0,
        V3: parseFloat(document.getElementById('input-v3').value) || 0,
        V4: parseFloat(document.getElementById('input-v4').value) || 0,
        V5: parseFloat(document.getElementById('input-v5').value) || 0,
        V14: parseFloat(document.getElementById('input-v14').value) || 0,
        V17: parseFloat(document.getElementById('input-v17').value) || 0
    };

    // Fill remaining V features with 0
    for (let i = 6; i <= 28; i++) {
        if (i !== 14 && i !== 17) {
            transaction[`V${i}`] = 0;
        }
    }

    try {
        const response = await fetch(`${API_BASE}/api/detect/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(transaction)
        });

        const result = await response.json();
        displayDetectionResult(result);
    } catch (error) {
        showToast('Detection failed: ' + error.message, 'error');
    }
}

function displayDetectionResult(result) {
    const container = document.getElementById('detection-result');
    const status = document.getElementById('result-status');
    const probability = document.getElementById('result-probability');
    const models = document.getElementById('result-models');

    container.classList.remove('hidden');

    // Status
    status.className = 'result-status ' + (result.is_fraud ? 'fraud' : 'safe');
    status.querySelector('.status-icon').innerHTML = result.is_fraud ?
        '<svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 15h2v2h-2v-2zm0-10h2v8h-2V7z"/></svg>' :
        '<svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>';
    status.querySelector('.status-text').textContent = result.is_fraud ?
        `FRAUD DETECTED (${result.risk_level})` : 'TRANSACTION SAFE';
    status.querySelector('.status-text').style.color = result.is_fraud ? '#ef4444' : '#10b981';

    // Probability
    probability.querySelector('.prob-value').textContent = (result.fraud_probability * 100).toFixed(1) + '%';

    // Model predictions
    models.innerHTML = '';
    if (result.model_predictions) {
        for (const [name, pred] of Object.entries(result.model_predictions)) {
            const isFraud = pred.predictions && pred.predictions[0] === 1;
            const prob = pred.probabilities ? (pred.probabilities[0] * 100).toFixed(1) : '--';

            models.innerHTML += `
                <div class="result-model">
                    <div class="model-name">${formatModelName(name)}</div>
                    <div class="model-pred ${isFraud ? 'fraud' : 'safe'}">
                        ${isFraud ? 'FRAUD' : 'SAFE'} (${prob}%)
                    </div>
                </div>
            `;
        }
    }
}

async function loadSampleTransaction() {
    try {
        const response = await fetch(`${API_BASE}/api/data/sample?n=1&fraud_only=false`);
        const data = await response.json();

        if (data.samples && data.samples.length > 0) {
            const sample = data.samples[0];
            document.getElementById('input-amount').value = sample.Amount?.toFixed(2) || '';
            document.getElementById('input-v1').value = sample.V1?.toFixed(2) || '';
            document.getElementById('input-v2').value = sample.V2?.toFixed(2) || '';
            document.getElementById('input-v3').value = sample.V3?.toFixed(2) || '';
            document.getElementById('input-v4').value = sample.V4?.toFixed(2) || '';
            document.getElementById('input-v5').value = sample.V5?.toFixed(2) || '';
            document.getElementById('input-v14').value = sample.V14?.toFixed(2) || '';
            document.getElementById('input-v17').value = sample.V17?.toFixed(2) || '';

            showToast('Loaded sample transaction', 'success');
        }
    } catch (error) {
        showToast('Failed to load sample', 'error');
    }
}

// ============================================
// Comparison
// ============================================

async function refreshComparison() {
    try {
        const response = await fetch(`${API_BASE}/api/metrics/comparison?sample_size=5000`);
        const data = await response.json();

        if (data.comparison) {
            const models = ['isolation_forest', 'autoencoder', 'xgboost'];
            const metrics = ['precision', 'recall', 'f1_score', 'auc_roc'];

            models.forEach((model, idx) => {
                if (data.comparison[model]) {
                    const m = data.comparison[model];
                    comparisonChart.data.datasets[idx].data = [
                        m.precision || 0,
                        m.recall || 0,
                        m.f1_score || 0,
                        m.auc_roc || 0
                    ];
                }
            });

            comparisonChart.update();
        }
    } catch (error) {
        console.error('Failed to refresh comparison:', error);
    }
}

// ============================================
// Utilities
// ============================================

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}

function formatModelName(name) {
    const names = {
        'isolation_forest': 'Isolation Forest',
        'autoencoder': 'Autoencoder',
        'xgboost': 'XGBoost'
    };
    return names[name] || name;
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.querySelector('.toast-message').textContent = message;
    toast.className = 'toast ' + type;

    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

// ============================================
// CSV Upload
// ============================================

// Initialize drag-drop
document.addEventListener('DOMContentLoaded', () => {
    const zone = document.getElementById('upload-zone');
    if (zone) {
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });

        zone.addEventListener('dragleave', () => {
            zone.classList.remove('dragover');
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.name.endsWith('.csv')) {
                processCSVFile(file);
            } else {
                showToast('Please drop a CSV file', 'error');
            }
        });
    }
});

function handleCSVUpload(event) {
    const file = event.target.files[0];
    if (file) {
        processCSVFile(file);
    }
}

async function processCSVFile(file) {
    showToast('Analyzing CSV...', 'info');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/api/upload/csv`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const result = await response.json();
        displayUploadResults(result);
        showToast(`Analyzed ${result.total_transactions} transactions`, 'success');
    } catch (error) {
        showToast('Upload failed: ' + error.message, 'error');
    }
}

async function analyzeSampleCSV() {
    showToast('Analyzing sample CSV...', 'info');

    try {
        const response = await fetch(`${API_BASE}/api/upload/analyze-sample`, {
            method: 'POST'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }

        const result = await response.json();
        displayUploadResults(result);
        showToast(`Analyzed ${result.total_transactions} sample transactions`, 'success');
    } catch (error) {
        showToast('Sample analysis failed: ' + error.message, 'error');
    }
}

function displayUploadResults(result) {
    const resultsDiv = document.getElementById('upload-results');
    resultsDiv.classList.remove('hidden');

    // Update summary stats
    document.getElementById('upload-total').textContent = result.total_transactions;
    document.getElementById('upload-fraud').textContent = result.fraud_detected;
    document.getElementById('upload-normal').textContent = result.normal_detected;

    if (result.summary_stats && result.summary_stats.detection_accuracy !== undefined) {
        document.getElementById('upload-accuracy').textContent = result.summary_stats.detection_accuracy + '%';
    } else {
        document.getElementById('upload-accuracy').textContent = '--';
    }

    // Update risk distribution
    const riskDiv = document.getElementById('risk-distribution');
    const risks = result.risk_distribution || {};
    riskDiv.innerHTML = `
        <div class="risk-bar critical">
            <span class="risk-count">${risks.CRITICAL || 0}</span>
            CRITICAL
        </div>
        <div class="risk-bar high">
            <span class="risk-count">${risks.HIGH || 0}</span>
            HIGH
        </div>
        <div class="risk-bar medium">
            <span class="risk-count">${risks.MEDIUM || 0}</span>
            MEDIUM
        </div>
        <div class="risk-bar low">
            <span class="risk-count">${risks.LOW || 0}</span>
            LOW
        </div>
        <div class="risk-bar minimal">
            <span class="risk-count">${risks.MINIMAL || 0}</span>
            MINIMAL
        </div>
    `;

    // Update results table
    const tbody = document.getElementById('results-table-body');
    tbody.innerHTML = '';

    result.transactions.forEach(tx => {
        const predClass = tx.predicted_fraud ? 'fraud-cell' : 'normal-cell';
        const predText = tx.predicted_fraud ? 'FRAUD' : 'Normal';
        const riskClass = tx.risk_level.toLowerCase();

        let actualCell = '--';
        if (tx.actual_class !== null) {
            const isCorrect = tx.correct;
            const icon = isCorrect ? '✓' : '✗';
            const iconClass = isCorrect ? 'correct-icon' : 'incorrect-icon';
            actualCell = `<span class="${iconClass}">${icon}</span> ${tx.actual_class === 1 ? 'Fraud' : 'Normal'}`;
        }

        tbody.innerHTML += `
            <tr>
                <td>${tx.index}</td>
                <td>$${tx.amount.toFixed(2)}</td>
                <td class="${predClass}">${predText}</td>
                <td>${(tx.fraud_probability * 100).toFixed(1)}%</td>
                <td><span class="risk-badge ${riskClass}">${tx.risk_level}</span></td>
                <td>${actualCell}</td>
            </tr>
        `;
    });
}
