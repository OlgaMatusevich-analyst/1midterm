// data-loader.js
// Parse local CSV, run EDA, one-hot encode categoricals, scale numerics, split into train/test.
// Removes noisy/constant features from model input: EmployeeNumber, StandardHours, EmployeeCount, Over18 (as feature).

export class DataLoader {
  constructor(opts = {}) {
    this.log = opts.log || (() => {});
    this.headers = [];
    this.raw = [];
    this.labelKey = 'Attrition';
    this.attritionMap = { Yes: 1, No: 0 };

    // Numeric columns (cleaned: no EmployeeNumber/StandardHours/EmployeeCount)
    this.numCols = new Set([
      'Age','DailyRate','DistanceFromHome','Education',
      'EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction',
      'MonthlyIncome','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating',
      'RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
      'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
      'YearsSinceLastPromotion','YearsWithCurrManager'
    ]);

    // Categoricals (cleaned: no Over18 as a feature)
    this.catKnown = new Set([
      'BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'
    ]);

    this.catCols = [];
    this.encoders = {}; // { col: {value:index} }
    this.featureOrder = [];
    this.scaler = null; // { mean:[], std:[] }

    // Keep some meta fields for export (not used as features)
    this.metaFields = ['EmployeeNumber','JobRole','OverTime','YearsAtCompany','MonthlyIncome'];
    this.metaRows = []; // per original row index
  }

  async fromFile(file) {
    if (!file) throw new Error('No file selected.');
    const text = await this.#readFileText(file);
    const { headers, rows } = this.#parseCSV(text);

    if (!headers.includes(this.labelKey)) {
      throw new Error(`Missing required column: ${this.labelKey}`);
    }

    this.headers = headers;
    this.catCols = headers.filter(h => this.catKnown.has(h));
    const allowed = new Set([...this.numCols, ...this.catCols, this.labelKey, ...this.metaFields]);

    this.raw = rows.map(r => {
      const obj = {};
      for (const k of headers) if (allowed.has(k)) obj[k] = r[k];
      return obj;
    });

    // Cache meta per-row
    this.metaRows = this.raw.map(r => {
      const m = {};
      for (const f of this.metaFields) m[f] = r[f] ?? '';
      return m;
    });

    this.log(`Loaded ${this.raw.length} rows (${this.catCols.length} categorical, ${this.numCols.size} numeric).`);
    return this;
  }

  // ================= EDA =================
  eda() {
    if (!this.raw?.length) throw new Error('Load data first.');
    const n = this.raw.length;

    // class balance
    let pos = 0, neg = 0;
    for (const r of this.raw) (String(r[this.labelKey]) === 'Yes') ? pos++ : neg++;
    const balance = { positive: pos, negative: neg, rate: pos / Math.max(1, n) };

    // correlations for numerics (point-biserial via Pearson)
    const y = this.raw.map(r => (String(r[this.labelKey]) === 'Yes' ? 1 : 0));
    const corr = {};
    for (const col of this.numCols) {
      const x = this.raw.map(r => Number(r[col] ?? 0));
      corr[col] = this.#pearson(x, y);
    }
    const topCorr = Object.entries(corr).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,8);

    // categorical attrition rates for OverTime and JobRole
    const catRates = {};
    for (const col of ['OverTime','JobRole']) {
      if (!this.catCols.includes(col) && col !== 'JobRole') continue; // JobRole is in headers anyway
      const rates = {};
      for (const r of this.raw) {
        const k = r[col] ?? '';
        if (!rates[k]) rates[k] = { pos:0, total:0 };
        rates[k].pos += (String(r[this.labelKey])==='Yes') ? 1 : 0;
        rates[k].total += 1;
      }
      const out = Object.entries(rates).map(([k,v]) => ({ k, rate: v.pos/Math.max(1,v.total) }))
                                      .sort((a,b)=>b.rate-a.rate);
      catRates[col] = out;
    }

    return { balance, topCorr, catRates };
  }

  // ============== Tensors & Splits ==============
  prepareTensors({ testSplit = 0.2 }) {
    if (!this.raw?.length) throw new Error('Dataset not loaded.');

    // build encoders for categoricals
    this.#fitCategoricals(this.raw);

    // raw feature vectors + labels
    const feats = [];
    const labels = [];
    for (const r of this.raw) {
      const { v, y } = this.#rowToFeatures(r);
      feats.push(v); labels.push(y);
    }

    // deterministic shuffle and split (keep index mapping for export)
    const idx = this.#deterministicShuffleIndex(feats.length);
    const nTest = Math.max(1, Math.floor(feats.length * Math.min(Math.max(testSplit,0.05),0.9)));
    const nTrain = feats.length - nTest;

    const trainIdx = idx.slice(0, nTrain);
    const testIdx  = idx.slice(nTrain);

    let Xtr = trainIdx.map(i => feats[i]);
    let ytr = trainIdx.map(i => [labels[i]]);
    const Xte = testIdx.map(i => feats[i]);
    const yte = testIdx.map(i => [labels[i]]);

    // scale on train only
    this.#fitScaler(Xtr);
    Xtr = this.#applyScaler(Xtr);
    const XteS = this.#applyScaler(Xte);

    const xTrain = tf.tensor2d(Xtr);
    const yTrain = tf.tensor2d(ytr);
    const xTest  = tf.tensor2d(XteS);
    const yTest  = tf.tensor2d(yte);

    // meta for test rows (for CSV export)
    const testMeta = testIdx.map(i => this.metaRows[i]);

    this.log(`Prepared tensors: xTrain ${xTrain.shape}, yTrain ${yTrain.shape}, xTest ${xTest.shape}, yTest ${yTest.shape}`);
    return {
      xTrain, yTrain, xTest, yTest,
      testMeta,
      attritionMap: this.attritionMap,
      featureOrder: this.featureOrder.slice()
    };
  }

  // ------------- helpers -------------
  #readFileText(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('Failed to read file.'));
      reader.onload = () => resolve(String(reader.result));
      reader.readAsText(file);
    });
  }

  #parseCSV(text) {
    const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n').filter(Boolean);
    if (lines.length < 2) throw new Error('CSV has no data.');
    const headers = lines[0].split(',').map(h => h.trim());
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(',');
      if (parts.length !== headers.length) continue;
      const obj = {};
      for (let j = 0; j < headers.length; j++) obj[headers[j]] = parts[j].trim();
      rows.push(obj);
    }
    return { headers, rows };
  }

  #fitCategoricals(rows) {
    this.encoders = {};
    for (const c of this.catCols) {
      const set = new Set();
      for (const r of rows) set.add(r[c] ?? '');
      const cats = Array.from(set.values()).sort();
      const enc = {}; cats.forEach((v,i)=>enc[v]=i);
      this.encoders[c] = enc;
    }
    const order = [];
    for (const col of this.numCols) order.push(col);
    for (const c of this.catCols) {
      const enc = this.encoders[c];
      for (const v of Object.keys(enc)) order.push(`${c}__${v}`);
    }
    this.featureOrder = order;
  }

  #rowToFeatures(r) {
    const feat = [];
    for (const col of this.numCols) {
      const v = Number(r[col] ?? 0);
      feat.push(Number.isFinite(v) ? v : 0);
    }
    for (const c of this.catCols) {
      const enc = this.encoders[c];
      const vec = new Array(Object.keys(enc).length).fill(0);
      const idx = enc[r[c] ?? ''];
      if (Number.isInteger(idx)) vec[idx] = 1;
      feat.push(...vec);
    }
    const y = this.attritionMap[String(r[this.labelKey] || 'No')] ?? 0;
    return { v: feat, y };
  }

  #fitScaler(X) {
    const d = X[0]?.length || 0;
    const mean = new Array(d).fill(0);
    const std  = new Array(d).fill(0);
    const n = X.length || 1;

    for (let j=0;j<d;j++){
      let s=0,s2=0;
      for (let i=0;i<n;i++){ const v=X[i][j]; s+=v; s2+=v*v; }
      mean[j]=s/n; std[j]=Math.sqrt(Math.max(1e-9, s2/n - mean[j]*mean[j]));
    }
    this.scaler = { mean, std };
  }
  #applyScaler(X) {
    const { mean,std } = this.scaler;
    return X.map(row => row.map((v,j)=>(v-mean[j])/std[j]));
  }

  #deterministicShuffleIndex(n) {
    const a = Array.from({length:n}, (_,i)=>i);
    let seed = 1337;
    const rand = () => (seed = (seed*1664525 + 1013904223) % 2**32) / 2**32;
    for (let i=n-1;i>0;i--) { const j = Math.floor(rand()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; }
    return a;
  }

  #pearson(x, y) {
    const n = Math.min(x.length, y.length);
    let sx=0, sy=0, sxx=0, syy=0, sxy=0;
    for (let i=0;i<n;i++){ const xi=+x[i]; const yi=+y[i];
      sx+=xi; sy+=yi; sxx+=xi*xi; syy+=yi*yi; sxy+=xi*yi; }
    const cov = sxy/n - (sx/n)*(sy/n);
    const vx = sxx/n - (sx/n)*(sx/n);
    const vy = syy/n - (sy/n)*(sy/n);
    const denom = Math.sqrt(Math.max(vx*vy,1e-12));
    return cov/denom;
  }
}
