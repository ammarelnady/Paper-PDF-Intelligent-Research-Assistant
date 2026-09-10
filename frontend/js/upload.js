/**
 * Upload Page JS logic.
 */

document.addEventListener('DOMContentLoaded', () => {
  const uploadArea = document.getElementById('uploadArea');
  const fileInput = document.getElementById('fileInput');
  const statusDiv = document.getElementById('uploadStatus');
  const docList = document.getElementById('recentDocsList');

  // Load existing documents
  loadRecentDocuments();

  if (uploadArea && fileInput) {
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
      uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadArea.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        handleFileUpload(fileInput.files[0]);
      }
    });
  }

  async function handleFileUpload(file) {
    if (uploadArea?.classList.contains('is-processing')) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Please select a valid PDF file.');
      return;
    }

    if (statusDiv) {
      statusDiv.innerHTML = `
        <div class="upload-status-row">
          <div class="loader"></div>
          <span>Processing &amp; Indexing Paper... Please wait</span>
        </div>
      `;
      uploadArea?.classList.add('is-processing');
    }

    try {
      const result = await API.uploadPDF(file);
      window.location.href = `paper.html?id=${encodeURIComponent(result.document_id)}`;
    } catch (err) {
      if (statusDiv) {
        statusDiv.innerHTML = `<div class="upload-status-row upload-error">Error: ${escapeHtml(err.message)}</div>`;
      }
      uploadArea?.classList.remove('is-processing');
    }
  }

  async function loadRecentDocuments() {
    if (!docList) return;
    try {
      const docs = await API.listDocuments();
      if (!docs || docs.length === 0) {
        docList.innerHTML = `<p style="color:var(--text-muted); font-size:0.9rem;">No papers uploaded yet.</p>`;
        return;
      }
      docList.innerHTML = docs.map(d => `
        <div class="card" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem; padding:1rem 1.25rem;">
          <div>
            <h4 style="font-size:1.05rem; margin-bottom:0.25rem;">${escapeHtml(d.title || d.filename)}</h4>
            <span style="color:var(--text-muted); font-size:0.85rem;">${d.num_pages} Pages • ${d.num_chunks} Chunks • ${d.num_sections} Sections</span>
          </div>
          <div style="display:flex; gap:0.5rem;">
            <a href="paper.html?id=${encodeURIComponent(d.document_id)}" class="btn" style="background:rgba(255,255,255,0.05); color:var(--text-primary);">View Summary</a>
            <a href="chat.html?id=${encodeURIComponent(d.document_id)}" class="btn btn-primary">Research Chat</a>
          </div>
        </div>
      `).join('');
    } catch (err) {
      docList.innerHTML = `<p style="color:var(--text-muted); font-size:0.9rem;">Ready to upload first paper.</p>`;
    }
  }

  function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = value || '';
    return div.innerHTML;
  }
});
