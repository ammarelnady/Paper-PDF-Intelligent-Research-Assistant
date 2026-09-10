/**
 * API client wrapper for PaperLens AI.
 */

const API_BASE = window.location.origin.includes('http') && !window.location.origin.includes('null')
  ? `${window.location.origin}/api`
  : 'http://localhost:8000/api';

const API = {
  async uploadPDF(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/documents/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(err.detail || 'Failed to upload document');
    }
    return res.json();
  },

  async listDocuments() {
    const res = await fetch(`${API_BASE}/documents/`);
    if (!res.ok) throw new Error('Failed to fetch documents');
    return res.json();
  },

  async getSummary(documentId) {
    const res = await fetch(`${API_BASE}/documents/${documentId}/summary`);
    if (!res.ok) throw new Error('Failed to fetch summary');
    return res.json();
  },

  async getSections(documentId) {
    const res = await fetch(`${API_BASE}/documents/${documentId}/sections`);
    if (!res.ok) throw new Error('Failed to fetch sections');
    return res.json();
  },

  async getUnderstanding(documentId) {
    const res = await fetch(`${API_BASE}/documents/${documentId}/understanding`);
    if (!res.ok) throw new Error('Failed to fetch paper understanding');
    return res.json();
  },

  async getSuggestedQuestions(documentId) {
    const res = await fetch(`${API_BASE}/questions/${documentId}`);
    if (!res.ok) throw new Error('Failed to fetch suggested questions');
    return res.json();
  },

  async askQuery(query, documentId = null, strategy = null) {
    const res = await fetch(`${API_BASE}/query/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        document_id: documentId,
        strategy,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Query failed' }));
      throw new Error(err.detail || 'Query request failed');
    }
    return res.json();
  },
};
