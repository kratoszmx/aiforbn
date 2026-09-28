'use strict';
const nodes = Object.fromEntries([...document.querySelectorAll('[id]')].map(el => [el.id, el]));
const state = {records: [], electrolytes: [], sources: [], evaluation: null};
const labels = {substrate:'基材',bn_form:'BN 形態',loading_mg_cm2:'BN 載量（mg/cm²）',loading_basis:'載量依據',bn_weight_pct:'BN 質量比例（%）',binder:'黏結劑',bn_binder_ratio:'BN:黏結劑（x:1）',ratio_basis:'配比依據',solvent:'溶劑',sonication_hours:'超聲時間（小時）',stirring:'攪拌方式',stirring_hours:'攪拌時間（小時）',dry_hours:'乾燥時間（小時）',dry_temperature_c:'乾燥溫度（°C）',applicator_gap_um:'塗布器間隙（μm）',coating_sides:'塗布面數',electrolyte:'電解液',polymer_salt_ratio:'聚合物與鹽比例',test_temperature_c:'測量溫度（°C）',final_thickness:'製備後膜厚',ionic_conductivity:'離子導電率',preparation_outcome:'製備結果',water_contact_angle:'水接觸角'};
const words = {PP:'PP 聚丙烯',PE:'PE 聚乙烯',calcium_alginate:'CA 海藻酸鈣',cellulose:'纖維素',solid_PEO_PVDF:'PEO / PVDF 固態電解質',raw_BNNT:'未純化 BNNT',purified_BNNT:'純化 BNNT',BN_nanopowder:'BN 奈米粉',BN_flakes:'BN 薄片',hBN:'六方氮化硼',none:'無',overnight:'過夜',failed_pore_clogging:'製備失敗：孔道堵塞',LiTFSI_DOL_DME_LiNO3:'LiTFSI + LiNO₃ / DOL:DME',LiPF6_carbonates:'LiPF₆ / 碳酸酯'};
const methods = {random:'隨機依序試驗',nearest:'參考相近配方',linear:'線性回歸推薦',adaptive_forest:'逐輪更新模型推薦'};

function escapeHtml(value) {
  return String(value ?? '未報告').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function numeric(value, digits=2) {
  return value === null || value === undefined ? '未報告' : Number(value).toLocaleString('zh-HK', {maximumFractionDigits: digits});
}
function readable(value) {
  if (value === null || value === undefined) return '未報告';
  if (typeof value === 'boolean') return value ? '是' : '否';
  return words[value] ?? String(value);
}
async function api(path, body) {
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
});
nodes['bnnt-form'].elements.bn_form.addEventListener('change', () => {
  const form = nodes['bnnt-form'];
  const max = form.elements.bn_form.value === 'raw_BNNT' ? .3 : .5;
  form.elements.loading_mg_cm2.max = max;
  if (Number(form.elements.loading_mg_cm2.value) > max) form.elements.loading_mg_cm2.value = max;
  nodes['bnnt-range'].textContent = `可估算載量：${max===.3?'未純化':'純化'} BNNT 0.01–${max.toFixed(2)} mg/cm²。`;
});
nodes['electrolyte-form'].addEventListener('input', () => {
  const form = nodes['electrolyte-form'];
  const available = 100-Number(form.elements.ec_percent.value);
  form.elements.dmc_percent.max = available;
  nodes['emc-percent'].textContent = numeric(available-Number(form.elements.dmc_percent.value))+'%';
});
document.querySelectorAll('.prediction-form').forEach(form => form.addEventListener('submit', async event => {
  event.preventDefault();
  const button = form.querySelector('[type=submit]');
  button.disabled = true;
  nodes.prediction.textContent = '正在分析配方，請稍候…';
  try {
    const result = await api('/api/predict', recipeFromForm(form));
    if (result.status === 'needs_data') {
      nodes.prediction.innerHTML = '<strong>需要補充這組條件的資料</strong>'+result.reasons.map(x => `<p>${escapeHtml(x)}</p>`).join('');
      return;
    }
    const p = result.prediction;
    const match = p.matched_observation_mS_cm ?? p.matched_observation_um;
    nodes.prediction.innerHTML = `<span class="tag">${escapeHtml(p.kind)}${result.cached?' · 已有分析結果':''}</span><p>${escapeHtml(p.property)}</p><p class="metric">${p.value===null?'本次未提供數值':numeric(p.value,3)+' '+escapeHtml(p.unit)}</p><p>${escapeHtml(result.explanation)}</p>${p.conditions?`<p class="muted">${escapeHtml(p.conditions)}</p>`:''}${match!==null&&match!==undefined?`<p>同配方文獻實測：${numeric(match,3)} ${escapeHtml(p.unit)}</p>`:''}<a href="${escapeHtml(p.source_url)}" target="_blank" rel="noopener">查看依據的研究</a>`;
  } catch (error) { nodes.prediction.textContent = error.message; }
  finally { button.disabled = false; }
}));

function renderWorkflow() {
  if (!state.evaluation?.summaries) { nodes['workflow-result'].textContent = '比較資料尚未載入。'; return; }
  const report = state.evaluation;
  const selected = report.summaries.find(x => x.target_mS_cm === Number(nodes['target-conductivity'].value));
  const baseline = nodes.baseline.value;
  const before = selected.methods.find(x => x.method === baseline);
  const after = selected.methods.find(x => x.method === 'adaptive_forest');
  const comparison = selected.comparisons.find(x => x.baseline === baseline);
  nodes['workflow-result'].innerHTML = `<span class="tag">公開資料回放 · 找到 ${selected.target_hits} 個達標配方</span><div class="comparison"><div><p>${methods[baseline]}</p><p class="metric">${numeric(before.mean_experiments)} 次</p></div><div><p>逐輪更新模型推薦</p><p class="metric">${numeric(after.mean_experiments)} 次</p></div></div><p class="gain">平均少做 <strong>${numeric(comparison.saved_experiments)} 次</strong>，實驗次數減少 <strong>${numeric(comparison.reduction_pct,1)}%</strong></p>`;
  nodes['workflow-context'].textContent = `${report.raw_measurement_count} 筆測量合併成 ${report.formulation_count} 種配方；${report.seeds} 種起始次序。每組方法共用前 ${report.initial_experiments} 個起始配方，達標即可提前結束；全部已做實驗都計入次數。這個目標共有 ${selected.eligible_formulations} 種達標配方。`;
  nodes['workflow-methods'].innerHTML = selected.methods.map(x => `<tr><td>${methods[x.method]}</td><td>${numeric(x.mean_experiments)} 次</td><td>${x.min_experiments}–${x.max_experiments} 次</td></tr>`).join('');
  const [lo,hi] = comparison.paired_start_bootstrap_95_pct;
  nodes['workflow-interval'].textContent = `按起始次序重抽樣的 95% 節省比例區間：${numeric(lo,1)}% 至 ${numeric(hi,1)}%。${lo<=0?'這項比較仍未顯示明確優勢。':'這組資料中，改善在不同起始次序下仍可見。'}區間只反映此資料池的起始次序差異。`;
  const hours = nodes['hours-per-experiment'].value;
  nodes['time-saved'].textContent = hours !== '' && nodes['hours-per-experiment'].validity.valid ? `若每次實驗投入 ${numeric(Number(hours))} 工時，這組回放差異相當於平均少 ${numeric(comparison.saved_experiments*Number(hours))} 工時的實驗工作量。日曆天數另受並行設備與等待時間影響。` : '填入工時後，可換算這組回放結果對應的工作量。';
}
for (const id of ['target-conductivity','baseline','hours-per-experiment']) nodes[id].addEventListener('input', renderWorkflow);

function renderCatalogue() {
  const electrolyte = nodes['catalogue-type'].value === 'electrolyte';
  const query = nodes['record-search'].value.trim().toLowerCase();
  const all = electrolyte ? state.electrolytes : state.records;
  const rows = all.filter(row => (electrolyte ? `${row.candidate_id} ${row.salt_molality} ${solventText(row)}` : `${row.sample} ${Object.values(row.inputs).map(readable).join(' ')} ${Object.keys(row.observations).map(k=>labels[k]).join(' ')}`).toLowerCase().includes(query));
  nodes['record-count'].textContent = `顯示 ${rows.length} / ${all.length} 種配方`;
  nodes['csv-download'].href = '/api/download/'+(electrolyte?'electrolytes.csv':'records.csv');
  nodes['record-details'].hidden = true;
  if (!rows.length) { nodes['record-table'].textContent = '沒有符合搜尋條件的配方。'; return; }
  const headings = electrolyte ? ['配方編號','LiPF₆ 濃度','溶劑比例（質量）','文獻實測導電率','測量次數 / 溫度','來源'] : ['配方名稱','基材 / BN','BN 用量','黏結劑 / 溶劑','已發表結果','詳情'];
  const body = rows.map(row => {
    if (electrolyte) return `<tr><td>${escapeHtml(row.candidate_id)}</td><td>${numeric(row.salt_molality)} mol/kg</td><td>${solventText(row)}</td><td><strong>${numeric(row.conductivity_mS_cm,3)} mS/cm</strong></td><td>${row.repeat_measurements} 次<small>${numeric(row.temperature_min_c)}–${numeric(row.temperature_max_c)}°C</small></td><td><a href="https://doi.org/10.1038/s41467-022-32938-1" target="_blank" rel="noopener">原始研究</a></td></tr>`;
    const loading = row.inputs.loading_mg_cm2 != null ? numeric(row.inputs.loading_mg_cm2)+' mg/cm²' : row.inputs.bn_weight_pct != null ? numeric(row.inputs.bn_weight_pct)+' wt%' : '未報告';
    const observations = Object.entries(row.observations).map(([k,v]) => `${escapeHtml(labels[k]??k)}：${escapeHtml(readable(v.value))} ${escapeHtml(v.unit??'')}`).join('<br>');
    return `<tr><td><strong>${escapeHtml(row.sample)}</strong></td><td>${escapeHtml(readable(row.inputs.substrate))}<small>${escapeHtml(readable(row.inputs.bn_form))}</small></td><td>${loading}</td><td>${escapeHtml(readable(row.inputs.binder))}<small>${escapeHtml(readable(row.inputs.solvent))}</small></td><td>${observations||'保留製備配方；原文未提供可抽取數值'}</td><td><button type="button" data-record="${escapeHtml(row.record_id)}">展開</button></td></tr>`;
  }).join('');
  nodes['record-table'].innerHTML = `<table><thead><tr>${headings.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>`;
}
for (const id of ['catalogue-type','record-search']) nodes[id].addEventListener('input', renderCatalogue);
nodes['record-table'].addEventListener('click', event => {
  const button = event.target.closest('[data-record]');
  if (!button) return;
  const row = state.records.find(x => x.record_id === button.dataset.record);
  const source = state.sources.find(x => x.source_id === row.source_id);
  nodes['record-details'].hidden = false;
  nodes['record-details'].innerHTML = `<h3>${escapeHtml(row.sample)}</h3><p><a href="${escapeHtml(source.url)}" target="_blank" rel="noopener">查看研究原文</a></p><div class="table-scroll"><table><tbody>${Object.entries(row.inputs).map(([k,v])=>`<tr><th>${escapeHtml(labels[k]??k)}</th><td>${escapeHtml(readable(v))}</td></tr>`).join('')}</tbody></table></div><p>${escapeHtml(row.notes)}</p><details><summary>核對原文段落</summary><div class="actions">${row.evidence_ids.map((id,index)=>`<button type="button" data-evidence="${escapeHtml(id)}">原文 ${index+1}</button>`).join('')}</div><div id="evidence-output"></div></details>`;
  nodes['record-details'].scrollIntoView({block:'nearest',behavior:'smooth'});
});
nodes['record-details'].addEventListener('click', async event => {
  const button = event.target.closest('[data-evidence]');
  if (!button) return;
  const output = document.getElementById('evidence-output');
  try { const item = await api('/api/evidence/'+encodeURIComponent(button.dataset.evidence)); output.innerHTML = `<p>${escapeHtml(item.locator)}</p><div class="evidence">${escapeHtml(item.text)}</div>`; }
  catch (error) { output.textContent = error.message; }
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
  try {
    const rows = [...nodes['observed-inputs'].querySelectorAll('.observed-row')];
    const entries = rows.map(row=>[row.querySelector('select').value,Number(row.querySelector('input').value)]);
    if (new Set(entries.map(x=>x[0])).size !== entries.length) throw Error('每個已測配方只需列出一次。');
    nodes['plan-result'].textContent = '正在安排下一輪…';
    const result = await api('/api/predict', {task:'experiment_plan',observations:Object.fromEntries(entries)});
    const recommended = result.recommendations.map(item => {
      const row = state.electrolytes.find(r=>r.candidate_id===item.candidate_id);
      return `<tr><td>${row.candidate_id}</td><td>${numeric(row.salt_molality)} mol/kg</td><td>${solventText(row)}</td><td>${numeric(item.prediction_mS_cm,3)} mS/cm</td></tr>`;
    }).join('');
    nodes['plan-result'].innerHTML = `<p>使用 ${result.observations_used} 個已測配方。${escapeHtml(result.explanation)}</p><div class="table-scroll"><table><thead><tr><th>推薦配方</th><th>LiPF₆ 濃度</th><th>溶劑比例</th><th>模型估算導電率</th></tr></thead><tbody>${recommended}</tbody></table></div>${recommended?'':'<p>所有候選配方都已有測量結果。</p>'}`;
  } catch (error) { nodes['plan-result'].textContent = error.message; }
  finally { button.disabled = false; }
});

async function load() {
  const [overview,sources,records,electrolytes,evaluation] = await Promise.all([api('/api/overview'),api('/api/sources'),api('/api/records'),api('/api/electrolytes'),api('/api/evaluation')]);
  Object.assign(state, {sources,records,electrolytes,evaluation});
  nodes.summary.innerHTML = [[overview.record_count,'種 BN 相關配方'],[overview.electrolyte_formulations,'種液態電解液配方'],[overview.electrolyte_measurements,'筆電解液實測']].map(([n,t])=>`<div class="stat"><strong>${n}</strong>${t}</div>`).join('');
  nodes.privacy.textContent = overview.privacy;
  nodes['catalogue-note'].textContent = `目前收錄 ${overview.record_count} 種 BN 相關配方，來自 ${overview.study_count} 篇原文研究；另收錄 ${overview.electrolyte_formulations} 種液態電解液配方。${overview.catalogue_note}`;
  nodes['usage-limit'].textContent = `全站每 24 小時接受 ${overview.daily_model_limit} 次新分析；查資料和讀取已有分析結果不佔額度。`;
  nodes.expiry.textContent = overview.expires_at_utc ? '公開試用預計至 '+new Date(overview.expires_at_utc).toLocaleString('zh-HK',{timeZone:'Asia/Hong_Kong'})+'（香港時間）。' : '本機試用。';
  nodes['source-list'].innerHTML = sources.map(s=>`<div class="source"><a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.title)}</a><p class="muted">${s.access==='full_text_xml'?'已核對全文':s.access==='abstract_only'?'摘要線索':'背景線索'}</p></div>`).join('')+'<div class="source"><a href="https://doi.org/10.1038/s41467-022-32938-1" target="_blank" rel="noopener">Clio：公開電解液實測研究</a><p><a href="https://github.com/BattModels/Clio-NatCommData" target="_blank" rel="noopener">作者提供的原始測量資料</a></p></div>';
  renderWorkflow();
  renderCatalogue();
  [0,8,16,24,32].forEach(addObservation);
}
// Discard legacy invitation fragments without storing the old credential.
if (new URLSearchParams(location.hash.slice(1)).has('access')) history.replaceState(null,'',location.pathname);
load().catch(error => { nodes.summary.textContent = '資料載入未完成：'+error.message; });
