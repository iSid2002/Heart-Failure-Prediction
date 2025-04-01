// API endpoints
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5001'
    : 'https://heart-failure-prediction-backend.onrender.com';

// Chart instance
let predictionChart = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded');
    const form = document.getElementById('predictionForm');
    console.log('Found form:', form);
    
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
        console.log('Form submit handler attached');
    } else {
        console.error('Could not find form with id "predictionForm"');
    }
});

function getFormData() {
    try {
        const formData = {
            age: parseFloat(document.getElementById('age').value),
            sex: parseInt(document.getElementById('sex').value),
            cp: parseInt(document.getElementById('cp').value),
            trestbps: parseFloat(document.getElementById('trestbps').value),
            chol: parseFloat(document.getElementById('chol').value),
            fbs: parseInt(document.getElementById('fbs').value),
            restecg: parseInt(document.getElementById('restecg').value),
            thalach: parseFloat(document.getElementById('thalach').value),
            exang: parseInt(document.getElementById('exang').value),
            oldpeak: parseFloat(document.getElementById('oldpeak').value),
            slope: parseInt(document.getElementById('slope').value),
            ca: parseInt(document.getElementById('ca').value),
            thal: parseInt(document.getElementById('thal').value)
        };

        // Validate all fields are present and have valid values
        for (const [key, value] of Object.entries(formData)) {
            if (value === null || isNaN(value)) {
                throw new Error(`Please enter a valid value for ${key}`);
            }
        }

        console.log('Collected form data:', formData);
        return formData;
    } catch (error) {
        console.error('Error collecting form data:', error);
        throw error;
    }
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    try {
        // Show loading spinner
        document.querySelector('.loading').style.display = 'block';
        document.getElementById('result').style.display = 'none';
        
        const formData = getFormData();
        console.log('Sending form data:', formData);
        
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const responseText = await response.text();
        console.log('Raw response:', responseText);
        
        if (!response.ok) {
            throw new Error(`Prediction request failed with status: ${response.status}, message: ${responseText}`);
        }
        
        let result;
        try {
            result = JSON.parse(responseText);
        } catch (e) {
            console.error('Failed to parse response:', e);
            throw new Error('Invalid response format from server');
        }
        
        console.log('Prediction result:', result);
        
        // Hide loading spinner
        document.querySelector('.loading').style.display = 'none';
        
        // Show result
        const resultContainer = document.getElementById('result');
        resultContainer.style.display = 'block';
        resultContainer.className = 'result-container ' + (result.prediction === 1 ? 'high-risk' : 'low-risk');
        
        document.getElementById('resultMessage').textContent = result.message || 'Prediction completed';
        document.getElementById('probability').textContent = 
            `Probability: ${((result.probability || 0) * 100).toFixed(2)}%`;
        
        // Update visualization
        updateChart(result.probability || 0);

        // Add print button
        const printButton = document.createElement('button');
        printButton.className = 'btn btn-primary mt-3';
        printButton.textContent = 'Print Report';
        printButton.onclick = () => printReport(formData, result);
        resultContainer.appendChild(printButton);
        
    } catch (error) {
        console.error('Error details:', error);
        document.querySelector('.loading').style.display = 'none';
        document.getElementById('result').style.display = 'none';
        alert(error.message || 'An error occurred while making the prediction. Please try again.');
    }
}

function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('result').style.display = 'none';
    if (predictionChart) {
        predictionChart.destroy();
        predictionChart = null;
    }
}

function updateChart(probability) {
    try {
        if (predictionChart) {
            predictionChart.destroy();
        }

        const ctx = document.getElementById('predictionChart').getContext('2d');
        if (!ctx) {
            console.error('Could not get chart context');
            return;
        }

        predictionChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk', 'High Risk'],
                datasets: [{
                    data: [1 - probability, probability],
                    backgroundColor: ['#198754', '#dc3545'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error updating chart:', error);
    }
}

// Function to print the medical report
function printReport(formData, result) {
    // Create a new window for the report
    const printWindow = window.open('', '_blank');
    
    // Get current date and time
    const reportDate = new Date().toLocaleString();
    
    // Create the report content with medical report styling
    printWindow.document.write(`
        <html>
        <head>
            <title>Heart Disease Risk Assessment Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    padding: 20px;
                    max-width: 800px;
                    margin: 0 auto;
                }
                .header {
                    text-align: center;
                    border-bottom: 2px solid #333;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }
                .section {
                    margin-bottom: 20px;
                }
                .result {
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                    text-align: center;
                }
                .high-risk {
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                    color: #721c24;
                }
                .low-risk {
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    color: #155724;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: left;
                }
                th {
                    background-color: #f8f9fa;
                }
                .footer {
                    margin-top: 40px;
                    text-align: center;
                    font-size: 0.9em;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Heart Disease Risk Assessment Report</h1>
                <p>Report Generated: ${reportDate}</p>
            </div>

            <div class="section">
                <h2>Patient Information</h2>
                <table>
                    <tr><th>Age</th><td>${formData.age} years</td></tr>
                    <tr><th>Gender</th><td>${formData.sex === 1 ? 'Male' : 'Female'}</td></tr>
                    <tr><th>Blood Pressure</th><td>${formData.trestbps} mm Hg</td></tr>
                    <tr><th>Cholesterol</th><td>${formData.chol} mg/dl</td></tr>
                    <tr><th>Fasting Blood Sugar</th><td>${formData.fbs === 1 ? '> 120 mg/dl' : '≤ 120 mg/dl'}</td></tr>
                    <tr><th>Maximum Heart Rate</th><td>${formData.thalach} bpm</td></tr>
                    <tr><th>Exercise Induced Angina</th><td>${formData.exang === 1 ? 'Yes' : 'No'}</td></tr>
                    <tr><th>ST Depression</th><td>${formData.oldpeak}</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>Risk Assessment Results</h2>
                <div class="result ${result.prediction === 1 ? 'high-risk' : 'low-risk'}">
                    <h3>${result.message}</h3>
                    <p>Risk Probability: ${(result.probability * 100).toFixed(2)}%</p>
                </div>
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    <li>Regular check-ups with your healthcare provider</li>
                    <li>Maintain a healthy diet and exercise routine</li>
                    <li>Monitor blood pressure and cholesterol levels</li>
                    <li>Follow prescribed medications and treatments</li>
                </ul>
            </div>

            <div class="footer">
                <p>This report is generated for medical reference purposes.</p>
                <p>Please consult with your healthcare provider for professional medical advice.</p>
            </div>
        </body>
        </html>
    `);
    
    // Print the report
    printWindow.document.close();
    printWindow.focus();
    setTimeout(() => {
        printWindow.print();
    }, 250);
}