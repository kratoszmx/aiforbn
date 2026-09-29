// Presentation-only translations. Formula values, IDs and source excerpts stay intact.
export const languages = ['zh-TW', 'zh-CN', 'en'];
export let language = 'zh-CN';
try {
  const saved = localStorage.getItem('science-language');
  if (languages.includes(saved)) language = saved;
} catch { /* A language choice still works when storage is unavailable. */ }

const response = await fetch('/messages.tsv');
if (!response.ok) throw Error('Language data could not be loaded.');
const rows = (await response.text()).trim().split('\n').map(line => line.split('\t'));
const exact = new Map();
const templates = [];
for (const row of rows) {
  if (row.length !== languages.length) throw Error('Invalid language catalogue.');
  for (const value of row) {
    const keys = [...value.matchAll(/\{(\w+)\}/g)].map(m => m[1]);
    if (!keys.length) { exact.set(value, row); continue; }
    const parts = value.split(/\{\w+\}/g).map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    templates.push({pattern:new RegExp('^'+parts.join('(.+?)')+'$'), keys, row});
  }
}

export function t(value) {
  const text = String(value ?? '未報告');
  const core = text.trim();
  const index = languages.indexOf(language);
  let translated = exact.get(core)?.[index];
  if (translated === undefined) {
    for (const {pattern, keys, row} of templates) {
      const match = core.match(pattern);
      if (!match) continue;
      const values = Object.fromEntries(keys.map((key,i) => [key,match[i+1]]));
      translated = row[index].replace(/\{(\w+)\}/g, (_,key) => values[key]);
      break;
    }
  }
  return translated === undefined ? text : text.replace(core, () => translated);
}

export function localizePage(root=document.documentElement) {
  document.documentElement.lang = language;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    if (!node.parentElement.closest('script,style,[data-original]')) node.nodeValue = t(node.nodeValue);
  }
  for (const element of root.querySelectorAll('[aria-label],[placeholder]')) {
    for (const attr of ['aria-label','placeholder']) {
      if (element.hasAttribute(attr)) element.setAttribute(attr,t(element.getAttribute(attr)));
    }
  }
}

export function selectLanguage(value) {
  if (!languages.includes(value)) return;
  language = value;
  try { localStorage.setItem('science-language',value); } catch { /* Keep the page usable. */ }
  localizePage();
}
