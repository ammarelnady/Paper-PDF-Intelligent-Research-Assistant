(function () {
  const storageKey = 'paper-research-theme';
  const root = document.documentElement;

  function getPreferredTheme() {
    const saved = localStorage.getItem(storageKey);
    if (saved === 'dark' || saved === 'light') return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function updateButton(button, theme) {
    if (!button) return;
    const dark = theme === 'dark';
    button.textContent = dark ? '☀ Light mode' : '☾ Dark mode';
    button.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
    button.setAttribute('aria-pressed', String(dark));
  }

  function applyTheme(theme) {
    root.dataset.theme = theme;
    localStorage.setItem(storageKey, theme);
    updateButton(document.getElementById('themeToggle'), theme);
  }

  applyTheme(getPreferredTheme());

  document.addEventListener('DOMContentLoaded', function () {
    const button = document.getElementById('themeToggle');
    updateButton(button, root.dataset.theme);
    if (button) {
      button.addEventListener('click', function () {
        applyTheme(root.dataset.theme === 'dark' ? 'light' : 'dark');
      });
    }
  });
})();
