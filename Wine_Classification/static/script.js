
// Sample row from your dataset - Class 0
const sampleData = {
    alcohol: 14.23, malic_acid: 1.71, ash: 2.43, alcalinity_of_ash: 15.6,
    magnesium: 127.0, total_phenols: 2.8, flavanoids: 3.06, nonflavanoid_phenols: 0.28,
    proanthocyanins: 2.29, color_intensity: 5.64, hue: 1.04, od280_od315_of_diluted_wines: 3.92,
    proline: 1065.0
}

// Mean values of wine dataset
const meanData = {
    alcohol: 13.0, malic_acid: 2.34, ash: 2.36, alcalinity_of_ash: 19.49,
    magnesium: 99.54, total_phenols: 2.29, flavanoids: 2.03, nonflavanoid_phenols: 0.36,
    proanthocyanins: 1.59, color_intensity: 5.06, hue: 0.96, od280_od315_of_diluted_wines: 2.61,
    proline: 746.89
}

function fillForm(data) {
    for (const [key, value] of Object.entries(data)) {
        document.querySelector(`input[name="${key}"]`).value = value;
    }
}

document.getElementById('loadSample').addEventListener('click', () => fillForm(sampleData));
document.getElementById('loadMean').addEventListener('click', () => fillForm(meanData));
document.getElementById('clearForm').addEventListener('click', () => document.getElementById('wineForm').reset());document.getElementById('wineForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => data[key] = parseFloat(value));

    const resultDiv = document.getElementById('result');
    resultDiv.classList.remove('hidden');
    resultDiv.innerHTML = "Predicting...";

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        resultDiv.innerHTML = `
            <h3>Prediction Result</h3>
            <p><b>Class:</b> ${result.predicted_label}</p>
            <p><b>Confidence:</b> ${result.confidence}%</p>
        `;
    } catch (error) {
        resultDiv.innerHTML = "Error: " + error;
    }
});