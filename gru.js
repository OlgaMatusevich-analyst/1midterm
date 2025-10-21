// gru.js
// GRU-based binary classifier for Attrition prediction in TF.js.
// Input shape: [N, 1, F] (single timestep). Output: Dense(sigmoid).

export class GRUClassifier {
  constructor(opts = {}) {
    this.log = opts.log || (() => {});
    this.model = null;
    this.inputShape = null; // [timesteps=1, features]
  }

  build({ timesteps = 1, features, units = 64, layers = 1, lr = 1e-3 }) {
    if (!Number.isInteger(features) || features < 1) throw new Error(`Invalid features: ${features}`);
    if (!Number.isInteger(timesteps) || timesteps < 1) throw new Error(`Invalid timesteps: ${timesteps}`);

    this.dispose();
    this.inputShape = [timesteps, features];

    const m = tf.sequential();
    const U = Math.max(8, units|0);
    const L = Math.max(1, layers|0);

    m.add(tf.layers.gru({
      units: U,
      inputShape: this.inputShape,
      returnSequences: L > 1,
      dropout: 0.1,
      recurrentDropout: 0.0,
      kernelInitializer: 'heNormal'
    }));
    for (let i = 1; i < L; i++) {
      const isLast = i === L - 1;
      m.add(tf.layers.gru({
        units: U,
        returnSequences: !isLast,
        dropout: 0.1,
        kernelInitializer: 'heNormal'
      }));
    }
    m.add(tf.layers.dense({ units: 64, activation: 'relu', kernelInitializer: 'heNormal' }));
    m.add(tf.layers.dropout({ rate: 0.2 }));
    m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    const optimizer = tf.train.adam(typeof lr === 'number' ? lr : 1e-3);
    m.compile({ optimizer, loss: 'binaryCrossentropy', metrics: ['binaryAccuracy'] });

    this.model = m;
    this.log(`Built GRU model: input [${timesteps}, ${features}], units=${U}, layers=${L}`);
    return this.model;
  }

  async fit({ xTrain, yTrain, epochs = 20, batchSize = 32, onEpoch, sampleWeight }) {
    if (!this.model) throw new Error('Build the model first.');
    const ep = Math.max(1, epochs|0), bs = Math.max(1, batchSize|0);
    return await this.model.fit(xTrain, yTrain, {
      epochs: ep, batchSize: bs, shuffle: true, sampleWeight,
      callbacks: { onEpochEnd: async (epoch, logs) => {
        this.log(`Epoch ${epoch+1}/${ep} — loss=${logs.loss?.toFixed(4)} acc=${(logs.binaryAccuracy??0).toFixed(4)}`);
        if (onEpoch) onEpoch(epoch, logs);
        await tf.nextFrame();
      }}
    });
  }

  predict(x) {
    if (!this.model) throw new Error('Model not built.');
    return tf.tidy(() => this.model.predict(x));
  }

  async evaluate({ xTest, yTest, threshold = 0.5 }) {
    const yProb = this.predict(xTest);
    const yPred = tf.tidy(() => yProb.greaterEqual(threshold).toInt());
    const yTrue = yTest.toInt();

    const tp = tf.tidy(() => tf.logicalAnd(yPred.equal(1), yTrue.equal(1)).sum().arraySync());
    const tn = tf.tidy(() => tf.logicalAnd(yPred.equal(0), yTrue.equal(0)).sum().arraySync());
    const fp = tf.tidy(() => tf.logicalAnd(yPred.equal(1), yTrue.equal(0)).sum().arraySync());
    const fn = tf.tidy(() => tf.logicalAnd(yPred.equal(0), yTrue.equal(1)).sum().arraySync());

    const acc = (tp + tn) / Math.max(1, tp + tn + fp + fn);
    const prec = tp / Math.max(1, tp + fp);
    const rec  = tp / Math.max(1, tp + fn);
    const f1   = 2 * prec * rec / Math.max(1e-9, (prec + rec));
    const auc  = await this.#rocAuc(yTest, yProb);

    yProb.dispose(); yPred.dispose(); yTrue.dispose();
    return { acc, prec, rec, f1, auc, cm: { tp, tn, fp, fn } };
  }

  async save(name='tfjs-attrition-gru') {
    if (!this.model) throw new Error('Model not built.');
    await this.model.save(`indexeddb://${name}`);
    this.log(`Saved to indexeddb://${name}`);
  }
  async load(name='tfjs-attrition-gru') {
    this.dispose();
    this.model = await tf.loadLayersModel(`indexeddb://${name}`);
    this.inputShape = this.model.inputs?.[0]?.shape?.slice(1);
    this.log(`Loaded model from indexeddb://${name}`);
    return this.model;
  }
  dispose(){ if (this.model){ this.model.dispose(); this.model=null; } }

  async #rocAuc(yTrue, yProb){
    const yt = (await yTrue.array()).map(r=>r[0]);
    const yp = (await yProb.array()).map(r=>r[0]);
    const pairs = yp.map((p,i)=>({p,y:yt[i]})).sort((a,b)=>b.p-a.p);
    let tp=0, fp=0; const P=yt.reduce((s,v)=>s+(v===1?1:0),0), N=yt.length-P;
    const roc=[{fpr:0,tpr:0}];
    for(const {y} of pairs){ if(y===1) tp++; else fp++; roc.push({fpr:fp/Math.max(1,N), tpr:tp/Math.max(1,P)}); }
    roc.push({fpr:1,tpr:1});
    let auc=0; for(let i=1;i<roc.length;i++){ const a=roc[i-1], b=roc[i]; auc+=(b.fpr-a.fpr)*(a.tpr+b.tpr)/2; }
    return Math.max(0, Math.min(1, auc));
  }
}
