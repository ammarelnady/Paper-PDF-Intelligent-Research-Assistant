/**
 * Chat page logic.
 * Sends queries to the backend, renders answers with route badges and citations.
 */

document.addEventListener('DOMContentLoaded', () => {
  const urlParams = new URLSearchParams(window.location.search);
  const docId = urlParams.get('id');
  const prefilledQuery = urlParams.get('q');

  const messagesEl = document.getElementById('chatMessages');
  const inputEl = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const strategySelect = document.getElementById('strategySelect');
  const chatDocBadge = document.getElementById('chatDocBadge');
  const paperNavLink = document.getElementById('paperNavLink');

  // Update nav links with document context
  if (docId) {
    if (chatDocBadge) {
      chatDocBadge.style.display = 'inline-block';
      chatDocBadge.textContent = 'Paper loaded';
    }
    if (paperNavLink) {
      paperNavLink.href = `paper.html?id=${encodeURIComponent(docId)}`;
    }
  }

  // Send on click or Enter key
  if (sendBtn) {
    sendBtn.addEventListener('click', () => sendMessage());
  }
  if (inputEl) {
    inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  // Auto-fill and send prefilled query from URL
  if (prefilledQuery && inputEl) {
    inputEl.value = prefilledQuery;
    setTimeout(() => sendMessage(), 300);
  }

  async function sendMessage() {
    const query = inputEl.value.trim();
    if (!query) return;

    // Render user message
    appendMessage('user', query);
    inputEl.value = '';
    inputEl.disabled = true;
    sendBtn.disabled = true;

    // Show loading indicator
    const loadingId = appendLoading();

    try {
      const strategy = strategySelect ? strategySelect.value || null : null;
      const result = await API.askQuery(query, docId, strategy);

      // Remove loading
      removeLoading(loadingId);

      // Render assistant response
      renderAssistantResponse(result);
    } catch (err) {
      removeLoading(loadingId);
      appendMessage('assistant', `Error: ${err.message}`, 'error');
    } finally {
      inputEl.disabled = false;
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  function appendMessage(role, text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatarDiv = document.createElement('div');
    if (role === 'user') {
      avatarDiv.className = 'avatar user-avatar';
      avatarDiv.textContent = 'U';
    } else {
      avatarDiv.className = 'avatar bot-avatar';
      avatarDiv.textContent = 'AI';
    }

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'bubble';
    if (type === 'error') {
      bubbleDiv.style.borderColor = 'rgba(239, 68, 68, 0.4)';
      bubbleDiv.style.color = '#fca5a5';
    }
    bubbleDiv.innerHTML = `<p>${escapeHtml(text)}</p>`;

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(bubbleDiv);
    messagesEl.appendChild(messageDiv);
    scrollToBottom();
  }

  function renderAssistantResponse(result) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'avatar bot-avatar';
    avatarDiv.textContent = 'AI';

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'bubble';

    // Route badge
    const routeClass = result.route === 'RAG' ? 'badge-rag' : result.route === 'WEB' ? 'badge-web' : 'badge-hybrid';
    let html = `<div class="route-pill"><span class="badge ${routeClass}">${result.route}</span><span style="color:var(--text-muted); font-size:0.8rem;">Confidence: ${(result.confidence * 100).toFixed(1)}%</span></div>`;

    // Answer text with line breaks
    html += `<div style="white-space: pre-wrap; line-height: 1.7;">${escapeHtml(result.answer)}</div>`;

    // Paper citations
    if (result.paper_citations && result.paper_citations.length > 0) {
      html += `<div style="margin-top: 1rem;"><span style="font-weight: 600; color: var(--accent-cyan); font-size: 0.85rem;">Paper Sources</span></div>`;
      result.paper_citations.forEach((c) => {
        html += `<div class="citation-card">
          <div class="citation-card-header">
            <span>Page ${c.page_number} | ${c.section}</span>
            <span style="color: var(--text-muted); font-size: 0.8rem;">Score: ${c.score.toFixed(4)}</span>
          </div>
          <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.35rem;">${escapeHtml(c.snippet)}</p>
        </div>`;
      });
    }

    // Web citations
    if (result.web_citations && result.web_citations.length > 0) {
      html += `<div style="margin-top: 1rem;"><span style="font-weight: 600; color: var(--accent-emerald); font-size: 0.85rem;">Web Sources</span></div>`;
      result.web_citations.forEach((w) => {
        html += `<div class="citation-card">
          <div class="citation-card-header">
            <a href="${escapeHtml(w.url)}" target="_blank" rel="noopener" style="color: var(--accent-cyan); text-decoration: none;">${escapeHtml(w.title)}</a>
            <span style="color: var(--text-muted); font-size: 0.8rem;">${escapeHtml(w.domain)}</span>
          </div>
          <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.35rem;">${escapeHtml(w.snippet)}</p>
        </div>`;
      });
    }

    bubbleDiv.innerHTML = html;
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(bubbleDiv);
    messagesEl.appendChild(messageDiv);
    scrollToBottom();
  }

  function appendLoading() {
    const id = 'loading-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message assistant';
    div.innerHTML = `
      <div class="avatar bot-avatar">AI</div>
      <div class="bubble" style="display:flex; align-items:center; gap:0.75rem;">
        <div class="loader"></div>
        <span style="color:var(--accent-cyan);">Analyzing and generating answer...</span>
      </div>
    `;
    messagesEl.appendChild(div);
    scrollToBottom();
    return id;
  }

  function removeLoading(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
  }
});
