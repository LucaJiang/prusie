/* Small offline conveniences; content and native navigation work without JS. */
const siteNavigation = document.querySelector('.site-nav');
if (siteNavigation) {
  const narrowScreen = window.matchMedia('(max-width: 850px)');
  const setNavigation = () => { siteNavigation.open = !narrowScreen.matches; };
  setNavigation();
  narrowScreen.addEventListener('change', setNavigation);
}

document.querySelectorAll('pre').forEach(pre => {
  pre.tabIndex = 0;
  pre.setAttribute('aria-label', 'Code example; scroll horizontally if needed');
  const wrap = document.createElement('div');
  wrap.className = 'code-block';
  pre.before(wrap);
  wrap.append(pre);
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'copy-code';
  button.textContent = 'Copy';
  button.setAttribute('aria-label', 'Copy code example');
  button.setAttribute('aria-live', 'polite');
  button.addEventListener('click', async () => {
    try {
      if (!navigator.clipboard) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(pre.textContent);
      button.textContent = 'Copied';
    } catch (_) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(pre);
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = 'Text selected';
    }
    setTimeout(() => { button.textContent = 'Copy'; }, 2000);
  });
  wrap.append(button);
});

/* Hints are shown only when a real local overflow exists, including late image loads. */
const scrollRegions = [...document.querySelectorAll('.table-wrap, .figure-wrap, div.arithmatex, pre')];
const overflowChecks = scrollRegions.map((region, index) => {
  const hint = document.createElement('p');
  hint.className = 'scroll-hint';
  hint.id = `overflow-note-${index}`;
  hint.hidden = true;
  hint.textContent = region.dataset.scrollLabel || (region.matches('div.arithmatex')
    ? 'Scroll horizontally to read the formula.' : 'Scroll horizontally to read the code.');
  region.after(hint);
  const descriptions = (region.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean);
  region.setAttribute('aria-describedby', [...descriptions, hint.id].join(' '));
  const update = () => { hint.hidden = !(region.clientWidth > 0 && region.scrollWidth > region.clientWidth + 1); };
  region.querySelectorAll('img').forEach(img => img.addEventListener('load', update));
  if (window.ResizeObserver) new ResizeObserver(update).observe(region);
  update();
  return update;
});
const updateOverflow = () => { overflowChecks.forEach(update => update()); };
window.addEventListener('resize', updateOverflow);
document.addEventListener('toggle', updateOverflow, true);
if (document.fonts) document.fonts.ready.then(updateOverflow);
