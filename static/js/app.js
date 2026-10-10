const $ = (selector) => document.querySelector(selector);
const fmt = (value, digits = 3) => Number(value).toFixed(digits);

function showError(message) {
  const notice = $('#notice');
  if (!notice) return;
  notice.textContent = message;
  notice.className = 'notice error';
}

function renderOverview(data) {
  const approved = data.class_counts.Approved || 0;
  const rejected = data.class_counts.Rejected || 0;
  const total = approved + rejected;
  $('#hero-model').textContent = data.selected_model;
  $('#hero-f1').textContent = fmt(data.selected_metrics.f1);
  $('#hero-auc').textContent = fmt(data.selected_metrics.roc_auc);
  $('#records').textContent = total.toLocaleString();
  $('#selected').textContent = data.selected_model;
  $('#approved').textContent = approved.toLocaleString();
  $('#approval-rate').textContent = `${(100 * approved / total).toFixed(1)}% of labeled applications`;
  $('#features').textContent = data.feature_count;
  $('#overview-accuracy').textContent = fmt(data.selected_metrics.accuracy);
  $('#overview-precision').textContent = fmt(data.selected_metrics.precision);
  $('#overview-recall').textContent = fmt(data.selected_metrics.recall);
  $('#overview-f1').textContent = fmt(data.selected_metrics.f1);
  $('#selection-reason').textContent = `${data.selection_reason} ${data.selected_model} achieved F1 ${fmt(data.selected_metrics.f1)} and ROC-AUC ${fmt(data.selected_metrics.roc_auc)} on the held-out set.`;
  const share = total ? approved / total : 0;
  $('#donut').style.background = `conic-gradient(#12a991 0 ${share * 360}deg,#dbe4eb ${share * 360}deg 360deg)`;
  $('#donut-total').textContent = total.toLocaleString();
  $('#legend').innerHTML = `<div><i class="approved-dot"></i>Approved<b>${approved.toLocaleString()}</b></div><div><i class="rejected-dot"></i>Rejected<b>${rejected.toLocaleString()}</b></div>`;
}

function renderPerformance(data) {
  $('#model-rows').innerHTML = data.models.map(model => `<tr class="${model.name === data.selected_model ? 'best-row' : ''}"><td>${model.name}${model.name === data.selected_model ? '<span class="best-label">SELECTED</span>' : ''}</td><td>${fmt(model.accuracy)}</td><td>${fmt(model.precision)}</td><td>${fmt(model.recall)}</td><td>${fmt(model.f1)}</td><td>${fmt(model.roc_auc)}</td></tr>`).join('');
  $('#selection-reason').textContent = `${data.selection_reason} ${data.selected_model}: F1 ${fmt(data.selected_metrics.f1)}, ROC-AUC ${fmt(data.selected_metrics.roc_auc)}.`;
  const matrix = data.selected_metrics.confusion_matrix;
  $('#matrix').innerHTML = `<table class="confusion-table"><thead><tr><th>Actual \ Predicted</th><th>Rejected</th><th>Approved</th></tr></thead><tbody><tr><th>Rejected</th><td>${matrix[0][0]}</td><td>${matrix[0][1]}</td></tr><tr><th>Approved</th><td>${matrix[1][0]}</td><td>${matrix[1][1]}</td></tr></tbody></table>`;
}

function renderImportance(data) {
  const importance = data.feature_importance || [];
  const max = Math.max(...importance.map(item => item.importance), 0.000001);
  $('#importance').innerHTML = importance.map((item, index) => `<div class="importance-row"><span class="rank">${String(index + 1).padStart(2, '0')}</span><span class="feature-name">${item.feature}</span><div class="track"><div class="fill" style="width:${Math.max(1, 100 * item.importance / max)}%"></div></div><b>${fmt(item.importance, 3)}</b></div>`).join('') || '<p>No measurable permutation importance was found.</p>';
}

function renderDataset(data) {
  const total = data.class_counts.Approved + data.class_counts.Rejected;
  $('#dataset-records').textContent = total.toLocaleString();
  $('#dataset-features').textContent = data.feature_count;
  $('#dataset-missing').textContent = data.missing_before.toLocaleString();
  $('#dataset-approval-rate').textContent = `${(100 * data.class_counts.Approved / total).toFixed(1)}%`;
  const distributions = data.distributions || {};
  const statistics = data.numeric_statistics || {};
  const maxOverall = Math.max(...Object.values(distributions).flatMap(bins => bins.map(bin => bin.count)), 1);
  $('#distribution-insights').innerHTML = Object.entries(distributions).map(([name, bins]) => {
    const stat = statistics[name] || {};
    return `<article class="distribution-card"><h3>${name.replaceAll('_', ' ')}</h3><p>Median ${Number(stat['50%']).toLocaleString(undefined, {maximumFractionDigits: 1})}</p><div class="histogram">${bins.map(bin => `<div class="hist-bin" title="${bin.label}: ${bin.count}"><i style="height:${Math.max(4, 132 * bin.count / maxOverall)}px"></i><small>${bin.label.split('–')[0]}</small></div>`).join('')}</div></article>`;
  }).join('');
  $('#numeric-summary').innerHTML = `<table><thead><tr><th>FEATURE</th><th>MEAN</th><th>MEDIAN</th><th>MINIMUM</th><th>MAXIMUM</th></tr></thead><tbody>${Object.entries(statistics).map(([name, stat]) => `<tr><td>${name.replaceAll('_', ' ')}</td><td>${Number(stat.mean).toLocaleString(undefined, {maximumFractionDigits: 1})}</td><td>${Number(stat['50%']).toLocaleString(undefined, {maximumFractionDigits: 1})}</td><td>${Number(stat.min).toLocaleString()}</td><td>${Number(stat.max).toLocaleString()}</td></tr>`).join('')}</tbody></table>`;
  $('#group-insights').innerHTML = Object.entries(data.approval_by || {}).map(([name, rows]) => `<article class="group-card"><h3>${name.replaceAll('_', ' ')}</h3>${rows.map(row => `<div class="group-row"><span>${row.label}<small> · ${row.count} applications</small></span><b>${(100 * row.rate).toFixed(1)}%</b></div>`).join('')}</article>`).join('');
}

function renderForm(data) {
  $('#fields').innerHTML = data.feature_schema.map(field => {
    let control;
    if (field.type === 'category') {
      control = `<select id="field-${field.name}" name="${field.name}" required>${field.options.map(value => `<option value="${value}" ${value === field.default ? 'selected' : ''}>${value}</option>`).join('')}</select>`;
    } else {
      control = `<input id="field-${field.name}" name="${field.name}" type="number" min="${field.min}" max="${field.max}" step="${field.step}" value="${field.default}" required>`;
    }
    return `<div class="field"><label for="field-${field.name}">${field.label}</label>${control}<small>Allowed: ${field.type === 'category' ? field.options.join(', ') : `${field.min}–${field.max}`}</small></div>`;
  }).join('');
}

function bindPredictionForm() {
  const form = $('#loan-form');
  if (!form) return;
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const button = $('#submit');
    const result = $('#result');
    button.disabled = true;
    button.textContent = 'Analyzing application…';
    result.className = 'result-card';
    result.innerHTML = '<span class="eyebrow dark">MODEL ESTIMATE</span><h3>Evaluating sample</h3><p>Applying the trained pipeline…</p>';
    try {
      const input = Object.fromEntries(new FormData(form).entries());
      const response = await fetch('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(input)});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Analysis failed.');
      result.className = `result-card ${data.approved ? 'approved' : 'rejected'}`;
      result.innerHTML = `<span class="eyebrow dark">MODEL ESTIMATE</span><h3>${data.approved ? 'Likely Approved' : 'Likely Rejected'}</h3><p>Prediction from ${data.model}</p><div class="confidence">${(data.probability * 100).toFixed(1)}% <small>model confidence for approval</small></div><p>${data.note}</p>`;
    } catch (error) {
      result.className = 'result-card';
      result.innerHTML = `<span class="eyebrow dark">ANALYSIS ISSUE</span><h3>Check the application</h3><p>${error.message}</p>`;
    } finally {
      button.disabled = false;
      button.innerHTML = 'Analyze application <b>→</b>';
    }
  });
}

async function init() {
  bindPredictionForm();
  try {
    const response = await fetch('/api/dashboard');
    const data = await response.json();
    if (!response.ok || !data.ready) {
      $('#model-status').innerHTML = '<i></i>Model setup needed';
      showError(data.message || 'Could not load trained model artifacts.');
      return;
    }
    $('#model-status').innerHTML = `<i></i>Model ready · ${data.selected_model}`;
    $('#model-status').classList.add('ready');
    const page = document.body.dataset.page;
    if (page === 'overview') renderOverview(data);
    if (page === 'application-analysis') renderForm(data);
    if (page === 'model-performance') renderPerformance(data);
    if (page === 'influencing-factors') renderImportance(data);
    if (page === 'dataset-insights') renderDataset(data);
  } catch (error) {
    $('#model-status').innerHTML = '<i></i>Server unavailable';
    showError('Could not connect to LoanIQ. Keep the Flask server running and reload this page.');
  }
}

init();
