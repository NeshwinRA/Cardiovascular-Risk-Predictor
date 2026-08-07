document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    const submitButton = document.querySelector("button[type=submit]");
    const overlay = document.getElementById("loadingOverlay");

    if (!form || !submitButton || !overlay) {
        return;
    }

    const errorPanel = document.createElement("section");
    errorPanel.className = "error-panel card";
    errorPanel.style.display = "none";
    form.parentNode.insertBefore(errorPanel, form);

    form.addEventListener("submit", function (event) {
        const requiredFields = [
            "age",
            "male",
            "currentSmoker",
            "cigsPerDay",
            "BPMeds",
            "prevalentStroke",
            "prevalentHyp",
            "diabetes",
            "totChol",
            "sysBP",
            "diaBP",
            "BMI",
            "heartRate",
            "glucose",
        ];
        const fieldLabels = {
            age: "Age",
            male: "Sex",
            currentSmoker: "Current smoker",
            cigsPerDay: "Cigs per day",
            BPMeds: "BP medication",
            prevalentStroke: "Previous stroke",
            prevalentHyp: "Hypertension",
            diabetes: "Diabetes",
            totChol: "Total cholesterol",
            sysBP: "Systolic BP",
            diaBP: "Diastolic BP",
            BMI: "BMI",
            heartRate: "Heart rate",
            glucose: "Glucose",
        };
        const errors = [];

        requiredFields.forEach((field) => {
            const input = document.querySelector(`[name=${field}]`);
            if (!input || input.value.trim() === "") {
                const label = fieldLabels[field] || field;
                errors.push(`${label} is required.`);
            }
        });

        if (errors.length > 0) {
            event.preventDefault();
            errorPanel.innerHTML = `
                <div class="error-title">Please fix the highlighted fields</div>
                <ul>${errors.map(error => `<li>${error}</li>`).join('')}</ul>
            `;
            errorPanel.style.display = "block";
            window.scrollTo({ top: errorPanel.offsetTop - 20, behavior: "smooth" });
            return;
        }

        errorPanel.style.display = "none";
        submitButton.textContent = "Analyzing...";
        submitButton.disabled = true;
        overlay.classList.add("active");
    });

    const resetButton = document.querySelector("button[type=reset]");
    if (resetButton) {
        resetButton.addEventListener("click", function () {
            errorPanel.style.display = "none";
        });
    }
});
