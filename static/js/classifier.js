/**
 * Classifier functionality for image upload and prediction
 */

document.addEventListener("DOMContentLoaded", function () {
  // Elements
  const form = document.getElementById("upload-form");
  const dropzone = document.getElementById("dropzone");
  const imageInput = document.getElementById("image-input");
  const browseBtn = document.getElementById("browse-btn");
  const dropzoneContent = document.getElementById("dropzone-content");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const removeBtn = document.getElementById("remove-btn");
  const submitBtn = document.getElementById("submit-btn");
  const resultSection = document.getElementById("result-section");
  const resultCard = document.getElementById("result-card");
  const resultIcon = document.getElementById("result-icon");
  const resultLabel = document.getElementById("result-label");
  const resultMessage = document.getElementById("result-message");
  const confidenceBar = document.getElementById("confidence-bar");
  const confidenceText = document.getElementById("confidence-text");
  const errorSection = document.getElementById("error-section");
  const errorMessage = document.getElementById("error-message");
  const newAnalysisBtn = document.getElementById("new-analysis-btn");

  // State
  let selectedFile = null;

  // Browse button click
  if (browseBtn) {
    browseBtn.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      imageInput.click();
    });
  }

  // Dropzone click
  if (dropzone) {
    dropzone.addEventListener("click", function (e) {
      if (e.target === dropzone || e.target.closest(".dropzone-content")) {
        imageInput.click();
      }
    });

    // Drag and drop events
    dropzone.addEventListener("dragover", function (e) {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", function (e) {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", function (e) {
      e.preventDefault();
      dropzone.classList.remove("dragover");

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleFile(files[0]);
      }
    });
  }

  // File input change
  if (imageInput) {
    imageInput.addEventListener("change", function () {
      if (this.files.length > 0) {
        handleFile(this.files[0]);
      }
    });
  }

  // Remove button click
  if (removeBtn) {
    removeBtn.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      resetUpload();
    });
  }

  // Form submit
  if (form) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      submitForm();
    });
  }

  // New analysis button
  if (newAnalysisBtn) {
    newAnalysisBtn.addEventListener("click", function () {
      resetAll();
      document
        .getElementById("upload-section")
        .scrollIntoView({ behavior: "smooth" });
    });
  }

  /**
   * Handle selected file
   */
  function handleFile(file) {
    // Validate file type
    const validTypes = ["image/jpeg", "image/png", "image/jpg"];
    if (!validTypes.includes(file.type)) {
      showError("Format file tidak didukung. Gunakan JPG atau PNG.");
      return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      showError("Ukuran file terlalu besar. Maksimum 10MB.");
      return;
    }

    selectedFile = file;
    showPreview(file);
    submitBtn.disabled = false;
    hideError();
  }

  /**
   * Show image preview
   */
  function showPreview(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImage.src = e.target.result;
      dropzoneContent.classList.add("d-none");
      previewContainer.classList.remove("d-none");
    };
    reader.readAsDataURL(file);
  }

  /**
   * Reset upload state
   */
  function resetUpload() {
    selectedFile = null;
    imageInput.value = "";
    previewImage.src = "";
    previewContainer.classList.add("d-none");
    dropzoneContent.classList.remove("d-none");
    submitBtn.disabled = true;
  }

  /**
   * Reset all state
   */
  function resetAll() {
    resetUpload();
    resultSection.classList.add("d-none");
    hideError();
  }

  /**
   * Submit form for classification
   */
  async function submitForm() {
    if (!selectedFile) {
      showError("Pilih gambar terlebih dahulu.");
      return;
    }

    // Show loading state
    setLoading(true);
    hideError();
    resultSection.classList.add("d-none");

    // Prepare form data
    const formData = new FormData();
    formData.append("image", selectedFile);

    // Get CSRF token
    const csrfToken = document.querySelector(
      "[name=csrfmiddlewaretoken]"
    ).value;

    try {
      const response = await fetch("/classify/", {
        method: "POST",
        body: formData,
        headers: {
          "X-CSRFToken": csrfToken,
        },
      });

      const data = await response.json();

      if (data.success) {
        showResult(data.prediction);
      } else {
        showError(data.error || "Terjadi kesalahan saat memproses gambar.");
      }
    } catch (error) {
      console.error("Error:", error);
      showError("Terjadi kesalahan koneksi. Silakan coba lagi.");
    } finally {
      setLoading(false);
    }
  }

  /**
   * Set loading state
   */
  function setLoading(isLoading) {
    const btnText = submitBtn.querySelector(".btn-text");
    const btnLoading = submitBtn.querySelector(".btn-loading");

    if (isLoading) {
      submitBtn.disabled = true;
      btnText.classList.add("d-none");
      btnLoading.classList.remove("d-none");
    } else {
      submitBtn.disabled = !selectedFile;
      btnText.classList.remove("d-none");
      btnLoading.classList.add("d-none");
    }
  }

  /**
   * Show classification result
   */
  function showResult(prediction) {
    const isAutistic = prediction.is_autistic;
    const confidence = prediction.confidence * 100;

    // Update result card styling
    resultCard.classList.remove("autistic", "non-autistic");
    resultCard.classList.add(isAutistic ? "autistic" : "non-autistic");

    // Update icon
    if (isAutistic) {
      resultIcon.innerHTML =
        '<i class="bi bi-exclamation-circle-fill text-warning" style="font-size: 5rem;"></i>';
    } else {
      resultIcon.innerHTML =
        '<i class="bi bi-check-circle-fill text-success" style="font-size: 5rem;"></i>';
    }

    // Update label
    resultLabel.textContent = prediction.label.replace("_", " ");
    resultLabel.className = `fw-bold mb-2 ${
      isAutistic ? "text-warning" : "text-success"
    }`;

    // Update message
    resultMessage.textContent = prediction.message;

    // Update confidence bar
    const barClass = isAutistic ? "bg-warning" : "bg-success";
    confidenceBar.className = `progress-bar ${barClass}`;
    confidenceBar.style.width = "0%";

    // Animate confidence bar
    setTimeout(() => {
      confidenceBar.style.width = `${confidence}%`;
      confidenceText.textContent = prediction.confidence_percentage;
    }, 100);

    // Show result section
    resultSection.classList.remove("d-none");

    // Scroll to result
    setTimeout(() => {
      resultSection.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 200);
  }

  /**
   * Show error message
   */
  function showError(message) {
    errorMessage.textContent = message;
    errorSection.classList.remove("d-none");
  }

  /**
   * Hide error message
   */
  function hideError() {
    errorSection.classList.add("d-none");
  }
});
