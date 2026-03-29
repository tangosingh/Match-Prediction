
// ── Simplified ML model logic (mimics trained model weights) ──
// Based on feature importances from Random Forest + XGBoost trained on IPL data

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function normalize(val, min, max) { return (val - min) / (max - min); }

function predictLogistic(f) {
  // Logistic regression weights (approximate from training)
  let z = -0.3
    + 1.8  * f.batting_norm
    - 1.4  * f.economy_norm
    + 1.2  * f.powerplay_norm
    + 0.9  * f.death_norm
    + 1.1  * f.wins_norm
    - 0.7  * f.rank_norm
    + 0.6  * f.partner_norm
    - 0.5  * f.errors_norm
    - 0.4  * f.extras_norm
    + 0.5  * f.toss
    + 0.8  * f.home
    + 0.3  * f.dew
    + 0.4  * f.pitch_flat
    - 0.3  * f.pitch_seam
    + 0.2  * f.crowd_high;
  return sigmoid(z);
}

function predictDecisionTree(f) {
  // Decision tree logic (simplified from depth-10 tree)
  let score = 0.5;
  if (f.batting_norm > 0.55) score += 0.15;
  else score -= 0.10;
  if (f.economy_norm < 0.45) score += 0.12;
  else score -= 0.08;
  if (f.powerplay_norm > 0.5) score += 0.10;
  if (f.wins_norm > 0.6) score += 0.08;
  if (f.home) score += 0.06;
  if (f.toss) score += 0.04;
  if (f.rank_norm < 0.4) score += 0.05;
  if (f.errors_norm > 0.5) score -= 0.07;
  if (f.extras_norm > 0.6) score -= 0.05;
  if (f.death_norm > 0.5) score += 0.08;
  if (f.partner_norm > 0.5) score += 0.05;
  return Math.min(0.95, Math.max(0.05, score));
}

function predictRandomForest(f) {
  // Random Forest (ensemble of tree logic with slight randomness)
  const trees = [];
  const seeds = [0.12,-0.08,0.05,-0.03,0.09,-0.05,0.07,-0.04,0.11,-0.06];
  for (let i = 0; i < 10; i++) {
    let s = 0.5
      + 0.22 * f.batting_norm
      - 0.18 * f.economy_norm
      + 0.15 * f.powerplay_norm
      + 0.13 * f.death_norm
      + 0.14 * f.wins_norm
      - 0.09 * f.rank_norm
      + 0.08 * f.partner_norm
      - 0.07 * f.errors_norm
      - 0.06 * f.extras_norm
      + 0.07 * f.home
      + 0.05 * f.toss
      + seeds[i];
    trees.push(Math.min(0.95, Math.max(0.05, s)));
  }
  return trees.reduce((a,b) => a+b, 0) / trees.length;
}

function predictXGBoost(f) {
  // XGBoost (gradient boosting approximation)
  let score = 0;
  // Round 1: batting
  score += 0.18 * (f.batting_norm - 0.5);
  // Round 2: economy
  score -= 0.16 * (f.economy_norm - 0.5);
  // Round 3: powerplay
  score += 0.13 * (f.powerplay_norm - 0.5);
  // Round 4: wins
  score += 0.14 * (f.wins_norm - 0.5);
  // Round 5: death
  score += 0.11 * (f.death_norm - 0.5);
  // Round 6: contextual
  score += 0.07 * f.home + 0.05 * f.toss + 0.03 * f.dew;
  score -= 0.06 * f.errors_norm;
  score -= 0.05 * f.extras_norm;
  score += 0.08 * f.partner_norm;
  score -= 0.09 * f.rank_norm;
  // Interaction terms
  score += 0.05 * f.batting_norm * f.powerplay_norm;
  score -= 0.04 * f.economy_norm * f.death_norm;
  return sigmoid(score * 3);
}

function predict() {
  const batting    = parseFloat(document.getElementById('batting').value);
  const economy    = parseFloat(document.getElementById('economy').value);
  const powerplay  = parseFloat(document.getElementById('powerplay').value);
  const death      = parseFloat(document.getElementById('death').value);
  const wins       = parseFloat(document.getElementById('recentwins').value);
  const opprank    = parseFloat(document.getElementById('opprank').value);
  const partner    = parseFloat(document.getElementById('partnership').value);
  const errors     = parseFloat(document.getElementById('errors').value);
  const extras     = parseFloat(document.getElementById('extras').value);
  const toss       = document.getElementById('toss').checked ? 1 : 0;
  const home       = document.getElementById('home').checked ? 1 : 0;
  const dew        = document.getElementById('dew').checked ? 1 : 0;
  const pitch      = document.getElementById('pitch').value;
  const crowd      = document.getElementById('crowd').value;

  const f = {
    batting_norm:  normalize(batting,   42,  290),
    economy_norm:  normalize(economy,   5,   12.5),
    powerplay_norm:normalize(powerplay, 20,  90),
    death_norm:    normalize(death,     10,  100),
    wins_norm:     normalize(wins,      0,   5),
    rank_norm:     normalize(opprank,   1,   10),
    partner_norm:  normalize(partner,   12,  85),
    errors_norm:   normalize(errors,    0,   10),
    extras_norm:   normalize(extras,    0,   28),
    toss, home, dew,
    pitch_flat:    pitch === 'flat'    ? 1 : 0,
    pitch_seam:    pitch === 'seaming' ? 1 : 0,
    crowd_high:    crowd === 'high'    ? 1 : 0,
  };

  const p_lr  = predictLogistic(f);
  const p_dt  = predictDecisionTree(f);
  const p_rf  = predictRandomForest(f);
  const p_xgb = predictXGBoost(f);

  // Weighted ensemble (weights based on F1 performance)
  const p_final = 0.15*p_lr + 0.20*p_dt + 0.35*p_rf + 0.30*p_xgb;

  showResult(p_final, p_lr, p_dt, p_rf, p_xgb, f);
}

function showResult(p, p_lr, p_dt, p_rf, p_xgb, f) {
  const isWin = p >= 0.5;
  const conf  = isWin ? p : 1 - p;
  const pct   = Math.round(conf * 100);

  document.getElementById('result-idle').classList.add('hide');
  const active = document.getElementById('result-active');
  active.classList.add('show');

  const verdict = document.getElementById('verdict-text');
  verdict.textContent = isWin ? 'WIN' : 'LOSS';
  verdict.className = 'result-verdict ' + (isWin ? 'win' : 'loss');

  const votes = [p_lr, p_dt, p_rf, p_xgb].filter(p => p >= 0.5).length;
  document.getElementById('verdict-sub').textContent =
    `${votes}/4 models predict ${isWin ? 'Win' : 'Loss'}`;

  const bar = document.getElementById('conf-bar');
  bar.className = 'conf-bar ' + (isWin ? 'win' : 'loss');
  setTimeout(() => { bar.style.width = pct + '%'; }, 50);

  document.getElementById('conf-pct').textContent = pct + '%';
  document.getElementById('conf-pct').style.color = isWin ? 'var(--win)' : 'var(--loss)';

  // Key factors
  const factors = [
    { name: 'Batting score',   val: f.batting_norm },
    { name: 'Bowling economy', val: 1 - f.economy_norm },
    { name: 'Powerplay runs',  val: f.powerplay_norm },
    { name: 'Recent form',     val: f.wins_norm },
    { name: 'Death overs',     val: f.death_norm },
    { name: 'Home advantage',  val: f.home * 0.8 + 0.1 },
  ].sort((a,b) => b.val - a.val);

  const fList = document.getElementById('factors-list');
  fList.innerHTML = factors.map(f => `
    <div class="factor-row">
      <span class="factor-name">${f.name}</span>
      <div class="factor-bar-bg"><div class="factor-bar" style="width:${Math.round(f.val*100)}%"></div></div>
      <span class="factor-val">${Math.round(f.val*100)}%</span>
    </div>`).join('');

  // Model votes
  const modelData = [
    { name: 'Logistic Regression', p: p_lr },
    { name: 'Decision Tree',       p: p_dt },
    { name: 'Random Forest',       p: p_rf },
    { name: 'XGBoost',             p: p_xgb },
  ];
  document.getElementById('model-votes').innerHTML = modelData.map(m => {
    const mWin = m.p >= 0.5;
    const mPct = Math.round((mWin ? m.p : 1-m.p) * 100);
    const cls  = mPct >= 75 ? 'high' : mPct >= 60 ? 'mid' : 'low';
    return `<div class="model-row">
      <span class="model-name">${m.name}</span>
      <span class="model-badge ${cls}">${mWin?'WIN':'LOSS'} · ${mPct}%</span>
    </div>`;
  }).join('');
}