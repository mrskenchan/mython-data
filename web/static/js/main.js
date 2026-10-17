// Lógica de Interfaz OrangeWeb Data Mining

document.addEventListener("DOMContentLoaded", function () {
  // Inicialización de tooltips de Bootstrap
  var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });

  // Manejo de Drag and Drop en la zona de subida (upload.html)
  const dropzone = document.getElementById("csvDropzone");
  const fileInput = document.getElementById("csvFileInput");
  const fileInfo = document.getElementById("fileInfoDisplay");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());

    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect(fileInput.files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) {
        handleFileSelect(fileInput.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    if (fileInfo) {
      fileInfo.classList.remove("d-none");
      document.getElementById("selectedFileName").textContent = file.name;
      document.getElementById("selectedFileSize").textContent = (file.size / 1024).toFixed(1) + " KB";
      const submitBtn = document.getElementById("btnUploadSubmit");
      if (submitBtn) submitBtn.disabled = false;
    }
  }

  // Actualizadores de valores de sliders
  const sliders = document.querySelectorAll(".form-range-reactive");
  sliders.forEach((slider) => {
    const targetLabelId = slider.getAttribute("data-target-label");
    const label = document.getElementById(targetLabelId);
    if (label) {
      slider.addEventListener("input", (e) => {
        label.textContent = e.target.value;
      });
    }
  });
});
