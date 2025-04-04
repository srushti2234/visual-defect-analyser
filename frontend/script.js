document.addEventListener("DOMContentLoaded", function () {
    const imageInput = document.getElementById("imageInput");
    const previewImage = document.getElementById("previewImage");
    const uploadButton = document.getElementById("uploadButton");
    const resultsDiv = document.getElementById("results");
    const defectList = document.getElementById("defectList");

    let selectedFile = null;

    // ✅ Preview Image Before Upload
    imageInput.addEventListener("change", function (event) {
        const file = event.target.files[0];
        if (file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = function (e) {
                previewImage.src = e.target.result;
                previewImage.classList.remove("hidden");
            };
            reader.readAsDataURL(file);
        }
    });

    // ✅ Upload Image & Detect Defects
    uploadButton.addEventListener("click", function () {
        if (!selectedFile) {
            alert("Please select an image first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedFile);

        fetch("http://127.0.0.1:8000/predict-defect/", {  
            method: "POST",
            body: formData,
            headers: {
                "Accept": "application/json"
            },
        })
        .then(response => response.json())
        .then(data => {
            defectList.innerHTML = "";
            if (data.detected_defects && data.detected_defects.length > 0) {
                data.detected_defects.forEach(defect => {
                    const listItem = document.createElement("li");
                    listItem.textContent = `${defect.defect}: ${defect.confidence}%`;
                    defectList.appendChild(listItem);
                });
                resultsDiv.classList.remove("hidden");
            } else {
                alert("No defects detected.");
            }
        })
        .catch(error => {
            console.error("Error uploading image:", error);
            alert("Failed to upload image. Check backend console for details.");
        });
    });
});
