'use strict';
import {language, t, localizePage, selectLanguage} from './i18n.js';
const nodes = Object.fromEntries([...document.querySelectorAll('[id]')].map(el => [el.id, el]));
const state = {records: [], aqueous: [], electrolytes: [], sources: [], evaluation: null};
let pendingAnalyses = 0;
nodes.language.value = language;
nodes.language.addEventListener('change', () => {
  selectLanguage(nodes.language.value);
  if (nodes.prediction.dataset.language && nodes.prediction.dataset.language !== language) {
    nodes.prediction.textContent = t('重新執行估算，可取得目前語言的配方分析。');
    delete nodes.prediction.dataset.language;
  }
});
const colorPreference = window.matchMedia('(prefers-color-scheme: dark)');
let chosenTheme;
try { chosenTheme = localStorage.getItem('science-theme'); } catch { /* The switch also works without storage. */ }
if (!['light','dark'].includes(chosenTheme)) chosenTheme = null;
function renderTheme() {
  const dark = (chosenTheme ?? (colorPreference.matches?'dark':'light')) === 'dark';
  document.documentElement.dataset.theme = dark?'dark':'light';
  nodes['theme-toggle'].textContent = dark?'亮色模式':'暗色模式';
  nodes['theme-toggle'].setAttribute('aria-label', dark?'切換至亮色模式':'切換至暗色模式');
  nodes['theme-toggle'].setAttribute('aria-pressed', String(dark));
  localizePage();
}
nodes['theme-toggle'].addEventListener('click', () => {
  chosenTheme = document.documentElement.dataset.theme === 'dark'?'light':'dark';
  try { localStorage.setItem('science-theme',chosenTheme); } catch { /* Keep the current page usable. */ }
  renderTheme();
});
colorPreference.addEventListener('change', renderTheme);
renderTheme();
const labels = {substrate:'基材',bn_form:'BN 形態',loading_mg_cm2:'BN 載量（mg/cm²）',loading_basis:'載量依據',bn_weight_pct:'BN 質量比例（%）',binder:'黏結劑',bn_binder_ratio:'BN:黏結劑（x:1）',ratio_basis:'配比依據',solvent:'溶劑',sonication_hours:'超聲時間（小時）',stirring:'攪拌方式',stirring_hours:'攪拌時間（小時）',dry_hours:'乾燥時間（小時）',dry_temperature_c:'乾燥溫度（°C）',applicator_gap_um:'塗布器間隙（μm）',coating_sides:'塗布面數',electrolyte:'電解液',polymer_salt_ratio:'聚合物與鹽比例',test_temperature_c:'測量溫度（°C）',final_thickness:'製備後膜厚',ionic_conductivity:'離子導電率',preparation_outcome:'製備結果',water_contact_angle:'水接觸角'};
const words = {PP:'PP 聚丙烯',PE:'PE 聚乙烯',calcium_alginate:'CA 海藻酸鈣',cellulose:'纖維素',solid_PEO_PVDF:'PEO / PVDF 固態電解質',raw_BNNT:'未純化 BNNT',purified_BNNT:'純化 BNNT',BN_nanopowder:'BN 奈米粉',BN_flakes:'BN 薄片',hBN:'六方氮化硼',none:'無',overnight:'過夜',failed_pore_clogging:'製備失敗：孔道堵塞',LiTFSI_DOL_DME_LiNO3:'LiTFSI + LiNO₃ / DOL:DME',LiPF6_carbonates:'LiPF₆ / 碳酸酯'};
Object.assign(words, {water:'水', 'solid PEO-PVDF-LiTFSI':'固態 PEO-PVDF-LiTFSI', 'no BN added':'未添加 BN', 'Reported powder ratio; mass basis not explicitly stated.':'粉末配比；原文未明示質量依據', 'Reported wt%; denominator not specified':'wt%；原文未明示分母'});
const methods = {random:'隨機依序試驗',nearest:'參考相近配方',linear:'線性回歸推薦',adaptive_forest:'逐輪更新模型推薦'};

function escapeHtml(value, original=false) {
  return (original ? String(value ?? '') : t(value)).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function numeric(value, digits=2) {
  return value === null || value === undefined ? t('未報告') : Number(value).toLocaleString(language, {maximumFractionDigits: digits});
}
function readable(value) {
  if (value === null || value === undefined) return t('未報告');
  if (typeof value === 'boolean') return t(value ? '是' : '否');
  return t(words[value] ?? String(value));
}
async function api(path, body) {
  if (body && path === '/api/predict') path += '?language='+encodeURIComponent(language);
  const response = await fetch(path, body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {});
  let data;
  try { data = await response.json(); } catch { throw Error('服務暫未回應，請稍後重試。'); }
  if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : (response.status === 410 ? '本次公開試用已結束。' : response.status === 422 ? '參數不在目前任務範圍，請核對數值與條件。' : '服務未完成請求，請稍後重試。'));
  return data;
}
function solventText(row) {
  return `EC ${numeric(100*row.ec_fraction)}% / DMC ${numeric(100*(1-row.ec_fraction)*row.dmc_ratio)}% / EMC ${numeric(100*(1-row.ec_fraction)*(1-row.dmc_ratio))}%`;
}
function recipeFromForm(form) {
  const values = Object.fromEntries(new FormData(form));
  if (form.dataset.task === 'electrolyte_conductivity') {
    const ec = Number(values.ec_percent)/100;
    return {task:form.dataset.task,salt_molality:Number(values.salt_molality),ec_fraction:ec,dmc_ratio:Number(values.dmc_percent)/(100*(1-ec))};
  }
  return {task:form.dataset.task,...Object.fromEntries(Object.entries(values).map(([k,v]) => [k,k==='bn_form'?v:Number(v)]))};
}
nodes.task.addEventListener('change', () => {
  document.querySelectorAll('.prediction-form').forEach(form => { form.hidden = form.dataset.task !== nodes.task.value; });
  nodes.prediction.textContent = '輸入參數，開始這個任務的數值估算。';
  delete nodes.prediction.dataset.language;
  localizePage();
});
nodes['bnnt-form'].elements.bn_form.addEventListener('change', () => {
  const form = nodes['bnnt-form'];
  const max = form.elements.bn_form.value === 'raw_BNNT' ? .3 : .5;
  form.elements.loading_mg_cm2.max = max;
  if (Number(form.elements.loading_mg_cm2.value) > max) form.elements.loading_mg_cm2.value = max;
  nodes['bnnt-range'].textContent = `可估算載量：${max===.3?'未純化':'純化'} BNNT 0.01–${max.toFixed(2)} mg/cm²。`;
  localizePage();
});
nodes['electrolyte-form'].addEventListener('input', () => {
  const form = nodes['electrolyte-form'];
  const available = 100-Number(form.elements.ec_percent.value);
  form.elements.dmc_percent.max = available;
  nodes['emc-percent'].textContent = numeric(available-Number(form.elements.dmc_percent.value))+'%';
});
nodes['aqueous-form'].addEventListener('input', () => {
  nodes['filler-percent'].textContent = numeric(85-Number(nodes['aqueous-form'].elements.bn_volume_pct.value))+'%';
});
document.querySelectorAll('.prediction-form').forEach(form => form.addEventListener('submit', async event => {
  event.preventDefault();
  const button = form.querySelector('[type=submit]');
  button.disabled = true;
  nodes.task.disabled = true;
  nodes.language.disabled = ++pendingAnalyses > 0;
  delete nodes.prediction.dataset.language;
  const remote = form.dataset.task === 'bnnt_conductivity';
  const started = Date.now();
  const waitingText = t(remote?'正在分析配方，通常約需 20–60 秒；已有結果會較快顯示。':'正在計算，通常約需 1–3 秒。');
  nodes.prediction.textContent = waitingText;
  nodes.prediction.setAttribute('aria-busy','true');
  const timer = setInterval(() => {
    const elapsed = Math.floor((Date.now()-started)/1000);
    nodes.prediction.textContent = remote && elapsed>=60
      ? t(`仍在分析，已等待 ${elapsed} 秒；本次分析最多約 2 分鐘。`)
      : `${waitingText} ${t(`已等待 ${elapsed} 秒。`)}`;
  },1000);
  try {
    const result = await api('/api/predict', recipeFromForm(form));
    if (result.status === 'needs_data') {
      nodes.prediction.innerHTML = '<strong>需要補充這組條件的資料</strong>'+result.reasons.map(x => `<p>${escapeHtml(x)}</p>`).join('');
      return;
    }
    const p = result.prediction;
    if (remote) nodes.prediction.dataset.language = language;
    const match = p.matched_observation_mS_cm ?? p.matched_observation_um ?? p.matched_observation_mPa_s;
    const neighbours = p.nearby_cases ? `<h3>相近配方的已發表結果</h3><div class="table-scroll"><table><thead><tr><th>BN 體積比例</th><th>黏度</th><th>塗布結果</th></tr></thead><tbody>${p.nearby_cases.map(r=>`<tr><td>${numeric(r.bn_volume_pct)}%</td><td>${numeric(r.viscosity_mPa_s)} mPa·s</td><td>${escapeHtml(r.coating_result)}</td></tr>`).join('')}</tbody></table></div>` : '';
    nodes.prediction.innerHTML = `<span class="tag"><span>${escapeHtml(p.kind)}</span>${result.cached?' · <span>已有分析結果</span>':''}</span><p>${escapeHtml(p.property)}</p><p class="metric">${p.value===null?'本次未提供數值':numeric(p.value,3)+' '+escapeHtml(p.unit)}</p><p>${escapeHtml(result.explanation)}</p>${p.conditions?`<p class="muted">${escapeHtml(p.conditions)}</p>`:''}${match!==null&&match!==undefined?`<p>同配方文獻實測：${numeric(match,3)} ${escapeHtml(p.unit)}</p>`:''}${neighbours}<a href="${escapeHtml(p.source_url)}" target="_blank" rel="noopener">查看配方出處</a>`;
  } catch (error) { nodes.prediction.textContent = error.message; }
  finally { clearInterval(timer); button.disabled = false; nodes.task.disabled = false; nodes.language.disabled = --pendingAnalyses > 0; nodes.prediction.setAttribute('aria-busy','false'); localizePage(); }
}));

function renderWorkflow() {
  if (!state.evaluation?.summaries) { nodes['workflow-result'].textContent = '比較資料尚未載入。'; return; }
  const report = state.evaluation;
  const selected = report.summaries.find(x => x.target_mS_cm === Number(nodes['target-conductivity'].value));
  const baseline = nodes.baseline.value;
  const before = selected.methods.find(x => x.method === baseline);
  const after = selected.methods.find(x => x.method === 'adaptive_forest');
  const comparison = selected.comparisons.find(x => x.baseline === baseline);
  nodes['workflow-result'].innerHTML = `<span class="tag">公開資料回放 · 找到 ${selected.target_hits} 個達標配方</span><div class="comparison"><div><p>${methods[baseline]}</p><p class="metric">${numeric(before.mean_experiments)} <span class="metric-unit">次</span></p></div><div><p>逐輪更新模型推薦</p><p class="metric">${numeric(after.mean_experiments)} <span class="metric-unit">次</span></p></div></div><p class="gain">回放中平均少做 <strong>${numeric(comparison.saved_experiments)} 次</strong>，實驗次數減少 <strong>${numeric(comparison.reduction_pct,1)}%</strong></p>`;

}
for (const id of ['target-conductivity','baseline']) nodes[id].addEventListener('input', () => { renderWorkflow(); localizePage(); });

function renderCatalogue() {
  const electrolyte = nodes['catalogue-type'].value === 'electrolyte';
  const aqueous = nodes['catalogue-type'].value === 'aqueous';
  const query = nodes['record-search'].value.trim().toLowerCase();
  const all = electrolyte ? state.electrolytes : aqueous ? state.aqueous : state.records;
  const rows = all.filter(row => (electrolyte ? `${row.candidate_id} ${row.salt_molality} ${solventText(row)}` : aqueous ? `${t(row.sample)} ${t(row.formulation)} ${t(row.particle_size)} ${t(row.process)} ${t(row.coating_result)}` : `${row.sample} ${Object.values(row.inputs).map(readable).join(' ')} ${Object.keys(row.observations).map(k=>t(labels[k]??k)).join(' ')}`).toLowerCase().includes(query));
  nodes['record-count'].textContent = `顯示 ${rows.length} / ${all.length} 種配方`;
  nodes['csv-download'].href = '/api/download/'+(electrolyte?'electrolytes.csv':aqueous?'aqueous.csv':'records.csv');
  nodes['record-details'].hidden = true;
  if (!rows.length) { nodes['record-table'].textContent = '沒有符合搜尋條件的配方。'; return; }
  const headings = electrolyte ? ['配方編號','LiPF₆ 濃度','溶劑比例（質量）','文獻實測導電率','測量次數 / 溫度','來源'] : aqueous ? ['配方名稱','用料與配比','粒徑','文獻報告黏度','塗布結果','詳情'] : ['配方名稱','基材 / BN','BN 用量','黏結劑 / 溶劑','已發表結果','詳情'];
  const body = rows.map(row => {
    if (electrolyte) return `<tr><td>${escapeHtml(row.candidate_id)}</td><td>${numeric(row.salt_molality)} mol/kg</td><td>${solventText(row)}</td><td><strong>${numeric(row.conductivity_mS_cm,3)} mS/cm</strong></td><td>${row.repeat_measurements}<small>${numeric(row.temperature_min_c)}–${numeric(row.temperature_max_c)}°C</small></td><td><a href="https://doi.org/10.1038/s41467-022-32938-1" target="_blank" rel="noopener">原始研究</a></td></tr>`;
    if (aqueous) return `<tr><td><strong>${escapeHtml(row.sample)}</strong><small>${escapeHtml(row.source_title)}</small></td><td>${escapeHtml(row.formulation)}</td><td>${escapeHtml(row.particle_size)}</td><td>${row.viscosity_mPa_s===null?'膏狀，未列數值':numeric(row.viscosity_mPa_s)+' mPa·s'}</td><td>${escapeHtml(row.coating_result)}</td><td><button type="button" data-aqueous="${escapeHtml(row.record_id)}">展開</button></td></tr>`;
    const loading = row.inputs.loading_mg_cm2 != null ? numeric(row.inputs.loading_mg_cm2)+' mg/cm²' : row.inputs.bn_weight_pct != null ? numeric(row.inputs.bn_weight_pct)+' wt%' : '未報告';
    const observations = Object.entries(row.observations).map(([k,v]) => `${escapeHtml(labels[k]??k)}：${escapeHtml(readable(v.value))} ${escapeHtml(v.unit??'')}`).join('<br>');
    return `<tr><td><strong>${escapeHtml(row.sample)}</strong></td><td>${escapeHtml(readable(row.inputs.substrate))}<small>${escapeHtml(readable(row.inputs.bn_form))}</small></td><td>${loading}</td><td>${escapeHtml(readable(row.inputs.binder))}<small>${escapeHtml(readable(row.inputs.solvent))}</small></td><td>${observations||'保留製備配方；原文未提供可抽取數值'}</td><td><button type="button" data-record="${escapeHtml(row.record_id)}">展開</button></td></tr>`;
  }).join('');
  nodes['record-table'].innerHTML = `<table><thead><tr>${headings.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>`;
}
for (const id of ['catalogue-type','record-search']) nodes[id].addEventListener('input', () => { renderCatalogue(); localizePage(); });
nodes['record-table'].addEventListener('click', event => {
  const aqueousButton = event.target.closest('[data-aqueous]');
  if (aqueousButton) {
    const row = state.aqueous.find(r=>r.record_id===aqueousButton.dataset.aqueous);
    nodes['record-details'].hidden = false;
    nodes['record-details'].innerHTML = `<h3>${escapeHtml(row.sample)}</h3><p>${escapeHtml(row.formulation)}</p><p>${escapeHtml(row.process)}</p><p>${escapeHtml(row.coating_result)}</p><p>出處：<a href="${escapeHtml(row.source_url)}" target="_blank" rel="noopener">${escapeHtml(row.source_title)}</a></p><p>${escapeHtml(row.section_title)}</p>`;
    nodes['record-details'].scrollIntoView({block:'nearest',behavior:'smooth'});
    localizePage();
    return;
  }
  const button = event.target.closest('[data-record]');
  if (!button) return;
  const row = state.records.find(x => x.record_id === button.dataset.record);
  const source = state.sources.find(x => x.source_id === row.source_id);
  nodes['record-details'].hidden = false;
  nodes['record-details'].innerHTML = `<h3>${escapeHtml(row.sample)}</h3><p><a href="${escapeHtml(source.url)}" target="_blank" rel="noopener">查看研究原文</a></p><div class="table-scroll"><table><tbody>${Object.entries(row.inputs).map(([k,v])=>`<tr><th>${escapeHtml(labels[k]??k)}</th><td>${escapeHtml(readable(v))}</td></tr>`).join('')}</tbody></table></div><details><summary>核對原文段落</summary><div class="actions">${row.evidence_ids.map((id,index)=>`<button type="button" data-evidence="${escapeHtml(id)}">原文 ${index+1}</button>`).join('')}</div><div id="evidence-output"></div></details>`;
  nodes['record-details'].scrollIntoView({block:'nearest',behavior:'smooth'});
  localizePage();
});
nodes['record-details'].addEventListener('click', async event => {
  const button = event.target.closest('[data-evidence]');
  if (!button) return;
  const output = document.getElementById('evidence-output');
  try { const item = await api('/api/evidence/'+encodeURIComponent(button.dataset.evidence)); output.innerHTML = `<p data-original><strong>${escapeHtml(item.source_title,true)}</strong></p><p>${escapeHtml(item.section_title)}</p><p>原文（保留文獻語言）</p><div class="evidence" data-original>${escapeHtml(item.text,true)}</div>`; }
  catch (error) { output.textContent = error.message; }
  localizePage();
});

function addObservation(index) {
  const row = document.createElement('div');
  row.className = 'observed-row';
  row.innerHTML = `<label>已測配方<select name="candidate">${state.electrolytes.map(r=>`<option value="${r.candidate_id}">${r.candidate_id} · ${numeric(r.salt_molality)} mol/kg · ${solventText(r)}</option>`).join('')}</select></label><label>實測 mS/cm<input name="measurement" type="number" min="0" max="100" step="any" required></label><button type="button" aria-label="移除已測配方">移除</button>`;
  const select = row.querySelector('select'), input = row.querySelector('input');
  select.selectedIndex = index;
  input.value = state.electrolytes[index].conductivity_mS_cm;
  select.addEventListener('change', () => { input.value = state.electrolytes.find(r=>r.candidate_id===select.value).conductivity_mS_cm; });
  row.querySelector('button').addEventListener('click', () => row.remove());
  nodes['observed-inputs'].append(row);
  localizePage(row);
}
nodes['add-observation'].addEventListener('click', () => {
  const existing = [...nodes['observed-inputs'].querySelectorAll('select')].map(x=>x.value);
  const index = state.electrolytes.findIndex(r=>!existing.includes(r.candidate_id));
  if (index >= 0) addObservation(index);
});
nodes['plan-form'].addEventListener('submit', async event => {
  event.preventDefault();
  const button = nodes['plan-form'].querySelector('[type=submit]');
  button.disabled = true;
  nodes.language.disabled = ++pendingAnalyses > 0;
  try {
    const rows = [...nodes['observed-inputs'].querySelectorAll('.observed-row')];
    const entries = rows.map(row=>[row.querySelector('select').value,Number(row.querySelector('input').value)]);
    if (new Set(entries.map(x=>x[0])).size !== entries.length) throw Error('每個已測配方只需列出一次。');
    nodes['plan-result'].textContent = t('正在安排下一輪，通常約需 1–3 秒。');
    const result = await api('/api/predict', {task:'experiment_plan',observations:Object.fromEntries(entries)});
    const recommended = result.recommendations.map(item => {
      const row = state.electrolytes.find(r=>r.candidate_id===item.candidate_id);
      return `<tr><td>${row.candidate_id}</td><td>${numeric(row.salt_molality)} mol/kg</td><td>${solventText(row)}</td><td>${numeric(item.prediction_mS_cm,3)} mS/cm</td></tr>`;
    }).join('');
    nodes['plan-result'].innerHTML = `<p><span>使用 ${result.observations_used} 個已測配方。</span> <span>${escapeHtml(result.explanation)}</span></p><div class="table-scroll"><table><thead><tr><th>推薦配方</th><th>LiPF₆ 濃度</th><th>溶劑比例</th><th>模型估算導電率</th></tr></thead><tbody>${recommended}</tbody></table></div>${recommended?'':'<p>所有候選配方都已有測量結果。</p>'}`;
  } catch (error) { nodes['plan-result'].textContent = error.message; }
  finally { button.disabled = false; nodes.language.disabled = --pendingAnalyses > 0; localizePage(); }
});

async function load() {
  const [overview,sources,records,aqueous,electrolytes,evaluation] = await Promise.all([api('/api/overview'),api('/api/sources'),api('/api/records'),api('/api/aqueous'),api('/api/electrolytes'),api('/api/evaluation')]);
  Object.assign(state, {sources,records,aqueous,electrolytes,evaluation});
  nodes.summary.innerHTML = [[overview.aqueous_formulations,'種水性漿料與對照'],[overview.record_count,'種 BN 相關配方'],[overview.electrolyte_formulations,'種液態電解液配方']].map(([n,t])=>`<div class="stat"><strong>${n}</strong>${t}</div>`).join('');
  nodes['catalogue-note'].textContent = '按材料體系查找用料、製程及已發表結果；展開配方可查看對應文獻與章節。';
  nodes['usage-limit'].textContent = `全站每 24 小時接受 ${overview.daily_model_limit} 次新分析；查資料和讀取已有分析結果不佔額度。`;
  nodes.expiry.textContent = overview.expires_at_utc ? '公開試用預計至 '+new Date(overview.expires_at_utc).toLocaleString('zh-HK',{timeZone:'Asia/Hong_Kong'})+'（香港時間）。' : '本機試用。';
  renderWorkflow();
  renderCatalogue();
  [0,8,16,24,32].forEach(addObservation);
  localizePage();
}
// Discard legacy invitation fragments without storing the old credential.
if (new URLSearchParams(location.hash.slice(1)).has('access')) history.replaceState(null,'',location.pathname);
load().catch(error => { nodes.summary.textContent = t('資料載入未完成：')+t(error.message); });
