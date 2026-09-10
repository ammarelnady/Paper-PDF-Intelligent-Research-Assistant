/**
 * Paper Overview & Understanding Page JS.
 */

document.addEventListener('DOMContentLoaded', async () => {
  const urlParams = new URLSearchParams(window.location.search);
  const docId = urlParams.get('id');

  const titleEl = document.getElementById('paperTitle');
  const summaryEl = document.getElementById('paperSummary');
  const problemEl = document.getElementById('problemStatement');
  const methodEl = document.getElementById('methodology');
  const findingsEl = document.getElementById('findings');
  const limitationsEl = document.getElementById('limitations');
  const contributionsEl = document.getElementById('keyContributions');
  const sectionsEl = document.getElementById('sectionsList');
  const topicsEl = document.getElementById('topicsList');
  const conceptsEl = document.getElementById('conceptsList');
  const questionsEl = document.getElementById('suggestedQuestionsList');
  const chatLink = document.getElementById('chatNavLink');
  const chatActionLink = document.getElementById('chatActionLink');

  if (!docId) {
    if (titleEl) titleEl.innerText = 'No Paper Selected';
    return;
  }

  const chatHref = `chat.html?id=${encodeURIComponent(docId)}`;
  if (chatLink) chatLink.href = chatHref;
  if (chatActionLink) chatActionLink.href = chatHref;

  try {
    // Load document metadata for a useful page title.
    const documents = await API.listDocuments();
    const document = documents.find(item => item.document_id === docId);
    if (titleEl && document) {
      const titleSpan = titleEl.querySelector('span');
      if (titleSpan) titleSpan.textContent = document.title || document.filename || 'Paper Overview';
    }

    // 1. Fetch Summary
    const summaryData = await API.getSummary(docId);
    if (summaryEl) summaryEl.innerText = summaryData.summary;
    if (problemEl) problemEl.innerText = summaryData.problem_statement || 'N/A';
    if (methodEl) methodEl.innerText = summaryData.methodology || 'N/A';
    if (findingsEl) findingsEl.innerText = summaryData.findings || 'N/A';
    if (limitationsEl) limitationsEl.innerText = summaryData.limitations || 'N/A';

    if (contributionsEl && summaryData.key_contributions) {
      contributionsEl.innerHTML = summaryData.key_contributions
        .map(c => `<li style="margin-bottom:0.4rem;">${escapeHtml(c)}</li>`)
        .join('');
    }

    // 2. Fetch Sections
    const sectionsData = await API.getSections(docId);
    if (sectionsEl) {
      sectionsEl.innerHTML = sectionsData.map(s => `
        <div class="card" style="margin-bottom:0.5rem; padding:0.75rem 1rem; display:flex; justify-content:space-between; align-items:center;">
          <span style="font-weight:600;">${escapeHtml(s.title)}</span>
          <span style="color:var(--text-muted); font-size:0.85rem;">Pages ${s.start_page} – ${s.end_page}</span>
        </div>
      `).join('');
    }

    // 3. Fetch Understanding (Topics & Concepts)
    const understandData = await API.getUnderstanding(docId);
    if (topicsEl && understandData.topics) {
      topicsEl.innerHTML = understandData.topics.map(t => `
        <span class="tag" style="background:rgba(59,130,246,0.15); border-color:rgba(59,130,246,0.3); color:#93c5fd;">
          # ${escapeHtml(t.name)}
        </span>
      `).join('');
    }

    if (conceptsEl && understandData.concepts) {
      conceptsEl.innerHTML = understandData.concepts.map(c => `
        <span class="tag" style="background:rgba(139,92,246,0.15); border-color:rgba(139,92,246,0.3); color:#c4b5fd;">
          💡 ${escapeHtml(c.name)}
        </span>
      `).join('');
    }

    // 4. Fetch Suggested Questions
    const questionsData = await API.getSuggestedQuestions(docId);
    if (questionsEl && questionsData) {
      questionsEl.innerHTML = questionsData.map(q => `
        <div class="card" style="margin-bottom:0.6rem; padding:0.85rem 1rem; display:flex; justify-content:space-between; align-items:center; gap:1rem;">
          <div>
            <span class="badge badge-rag" style="font-size:0.7rem; margin-bottom:0.25rem;">${escapeHtml(q.category)}</span>
            <p style="font-size:0.95rem; font-weight:500;">${escapeHtml(q.question)}</p>
          </div>
          <a href="chat.html?id=${encodeURIComponent(docId)}&q=${encodeURIComponent(q.question)}" class="btn" style="background:rgba(255,255,255,0.08); color:var(--text-primary); padding:0.4rem 0.8rem; font-size:0.85rem;">
            Ask 💬
          </a>
        </div>
      `).join('');
    }

  } catch (err) {
    console.error('Failed to load paper details', err);
    if (titleEl && titleEl.querySelector('span')) titleEl.querySelector('span').textContent = 'Unable to load paper';
    if (summaryEl) summaryEl.textContent = `Could not load this paper: ${err.message}`;
  }
  function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = value || '';
    return div.innerHTML;
  }
});
