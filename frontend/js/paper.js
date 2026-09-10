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
  const paperMetaEl = document.getElementById('paperMeta');
  const summaryModelEl = document.getElementById('summaryModel');
  const addSectionBtn = document.getElementById('addSectionBtn');
  let sectionsDataCache = [];

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
      if (paperMetaEl) {
        paperMetaEl.textContent = `${document.num_pages || 0} pages · ${document.num_chunks || 0} chunks · ${document.num_sections || 0} sections`;
      }
    }

    // 1. Fetch Summary
    const summaryData = await API.getSummary(docId);
    if (summaryEl) summaryEl.innerText = summaryData.summary;
    if (summaryModelEl) summaryModelEl.textContent = summaryData.model_used || 'Structured analysis';
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
    sectionsDataCache = sectionsData;
    if (sectionsEl) renderOutline(sectionsData);
    if (addSectionBtn) {
      addSectionBtn.addEventListener('click', () => addCustomSubsection(sectionsData));
    }

    // 3. Fetch Understanding (Topics & Concepts)
    const understandData = await API.getUnderstanding(docId);
    if (topicsEl && understandData.topics) {
      topicsEl.innerHTML = understandData.topics.length ? understandData.topics.map(t => `
        <span class="tag topic-tag">
          # ${escapeHtml(t.name)}
        </span>
      `).join('') : '<span class="tag">No topics detected</span>';
    }

    if (conceptsEl && understandData.concepts) {
      conceptsEl.innerHTML = understandData.concepts.length ? understandData.concepts.map(c => `
        <span class="tag concept-tag">
          💡 ${escapeHtml(c.name)}
        </span>
      `).join('') : '<span class="tag">No concepts detected</span>';
    }

    // 4. Fetch Suggested Questions
    const questionsData = await API.getSuggestedQuestions(docId);
    if (questionsEl && questionsData) {
      questionsEl.innerHTML = questionsData.length ? questionsData.map(q => `
        <div class="question-item">
          <div>
            <span class="badge badge-rag" style="font-size:0.7rem; margin-bottom:0.25rem;">${escapeHtml(q.category)}</span>
            <p style="font-size:0.95rem; font-weight:500;">${escapeHtml(q.question)}</p>
          </div>
          <a href="chat.html?id=${encodeURIComponent(docId)}&q=${encodeURIComponent(q.question)}" class="btn" style="background:rgba(255,255,255,0.08); color:var(--text-primary); padding:0.4rem 0.8rem; font-size:0.85rem;">
            Ask 💬
          </a>
        </div>
      `).join('') : '<p class="empty-state">No suggested questions were generated.</p>';
    }

  } catch (err) {
    console.error('Failed to load paper details', err);
    if (titleEl && titleEl.querySelector('span')) titleEl.querySelector('span').textContent = 'Unable to load paper';
    if (summaryEl) summaryEl.textContent = `Could not load this paper: ${err.message}`;
  }

  function renderOutline(sections) {
    const custom = JSON.parse(localStorage.getItem(`paper-outline-${docId}`) || '[]');
    const allSections = sections.concat(custom);
    if (!allSections.length) {
      sectionsEl.innerHTML = '<p class="empty-state">No sections were detected.</p>';
      return;
    }
    sectionsEl.innerHTML = allSections.map((section, index) => `
      <details class="outline-item level-${Math.min(section.level || 1, 5)}" ${section.level === 1 ? 'open' : ''}>
        <summary>
          <span class="outline-title">${escapeHtml(section.title)}</span>
          <span class="outline-meta">Pages ${section.start_page || '—'} – ${section.end_page || '—'}</span>
        </summary>
        <div class="outline-actions">
          <span>${section.parent_section_id ? 'Nested subsection' : 'Top-level section'}</span>
          <button class="outline-add-child" type="button" data-index="${index}">＋ Add child</button>
        </div>
      </details>
    `).join('');
    sectionsEl.querySelectorAll('.outline-add-child').forEach(button => {
      button.addEventListener('click', () => addCustomSubsection(allSections[Number(button.dataset.index)]));
    });
  }

  function addCustomSubsection(parent) {
    const title = window.prompt(`Add a subsection under “${parent?.title || 'Paper'}”:`);
    if (!title || !title.trim()) return;
    const key = `paper-outline-${docId}`;
    const custom = JSON.parse(localStorage.getItem(key) || '[]');
    custom.push({
      section_id: `custom-${Date.now()}`,
      title: title.trim(),
      level: Math.min((parent?.level || 1) + 1, 8),
      parent_section_id: parent?.section_id || null,
      start_page: parent?.start_page || null,
      end_page: parent?.end_page || null,
    });
    localStorage.setItem(key, JSON.stringify(custom));
    renderOutline(sectionsDataCache);
  }
  function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = value || '';
    return div.innerHTML;
  }
});
