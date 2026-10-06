/* Optional conveniences; navigation and all content work without JavaScript. */
document.querySelectorAll('pre').forEach(pre => {
  pre.tabIndex = 0;
  pre.setAttribute('aria-label', 'Code example; scroll horizontally if needed');
  if (!navigator.clipboard) return;
  const wrap = document.createElement('div');
  wrap.className = 'code-block';
  pre.before(wrap);
  wrap.append(pre);
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'copy-code';
  button.textContent = 'Copy';
  button.setAttribute('aria-label', 'Copy code example');
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(pre.textContent);
      button.textContent = 'Copied';
    } catch (_) {
      button.textContent = 'Select text to copy';
    }
    setTimeout(() => { button.textContent = 'Copy'; }, 2000);
  });
  wrap.append(button);
});
