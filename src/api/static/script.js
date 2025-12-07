// src/api/static/script.js

document.addEventListener('DOMContentLoaded', () => {
    fetchVersion();
});

// Navigation Logic
function showPanel(panelId) {
    // Hide all panels
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));

    // Show selected
    document.getElementById(panelId).classList.add('active');
    document.getElementById(`btn-${panelId}`).classList.add('active');
}

// Fetch Version Info
async function fetchVersion() {
    try {
        const response = await fetch('/version');
        const data = await response.json();
        document.getElementById('version-display').innerHTML =
            `Code: ${data.code_version} <br> Model: ${data.model_version}`;
    } catch (error) {
        console.error('Error fetching version:', error);
    }
}

// Prediction Logic
async function makePrediction(event) {
    event.preventDefault();
    const form = document.getElementById('predict-form');
    const resultDiv = document.getElementById('prediction-result');

    // Create JSON object from form data
    // NOTE: Ensure these input names match your Pydantic "Person" schema
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Convert numeric strings to numbers if your schema requires it
    // Example: data.age = Number(data.age);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        resultDiv.style.display = 'block';
        resultDiv.innerHTML = `<strong>Prediction Result:</strong> ${JSON.stringify(result.predictions)}`;

    } catch (error) {
        resultDiv.style.display = 'block';
        resultDiv.style.backgroundColor = '#fef2f2';
        resultDiv.style.borderColor = '#ef4444';
        resultDiv.style.color = '#991b1b';
        resultDiv.textContent = 'Error making prediction. Check console.';
        console.error(error);
    }
}

// Monitoring Logic
function loadReport(endpoint) {
    const windowSize = document.getElementById('window-size').value || 3000;
    const iframe = document.getElementById('report-frame');

    // Set iframe src to the API endpoint
    // The browser will load the returned HTML file directly inside the frame
    iframe.src = `${endpoint}?window_size=${windowSize}`;
}