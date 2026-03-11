/**
 * model.js – TensorFlow.js neural network builder, trainer, and inference utils.
 */
import * as tf from '@tensorflow/tfjs';

export const ACTIVATIONS = ['relu', 'sigmoid', 'tanh', 'linear'];

// ─── Build model ──────────────────────────────────────────────────────────────
export function buildModel(hiddenLayers, learningRate, regularization, regRate) {
  const model = tf.sequential();

  function makeReg() {
    if (regularization === 'l1') return tf.regularizers.l1({ l1: regRate });
    if (regularization === 'l2') return tf.regularizers.l2({ l2: regRate });
    return null;
  }

  if (hiddenLayers.length === 0) {
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid', inputShape: [2] }));
  } else {
    hiddenLayers.forEach(([units, activation], idx) => {
      model.add(tf.layers.dense({
        units,
        activation,
        inputShape: idx === 0 ? [2] : undefined,
        kernelRegularizer: makeReg(),
      }));
    });
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  }

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

// ─── Training step ────────────────────────────────────────────────────────────
export async function trainEpoch(model, trainX, trainY, testX, testY, batchSize) {
  const xTrain = tf.tensor2d(trainX);
  const yTrain = tf.tensor2d(trainY, [trainY.length, 1]);
  const xTest  = tf.tensor2d(testX);
  const yTest  = tf.tensor2d(testY,  [testY.length,  1]);

  const history = await model.fit(xTrain, yTrain, {
    batchSize: Math.max(1, batchSize),
    epochs: 1,
    shuffle: true,
    validationData: [xTest, yTest],
    verbose: 0,
  });

  tf.dispose([xTrain, yTrain, xTest, yTest]);
  return {
    trainLoss: history.history.loss[0],
    testLoss:  history.history.val_loss[0],
  };
}

// ─── Output grid (50×50) ──────────────────────────────────────────────────────
const GRID_N = 50;

export function computeGrid(model) {
  return tf.tidy(() => {
    const n = GRID_N;
    const xs = [];
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        xs.push([-1.5 + (3 * i) / (n - 1), -1.5 + (3 * j) / (n - 1)]);
      }
    }
    const inputT = tf.tensor2d(xs);
    const outputData = Array.from(model.predict(inputT).dataSync());

    const gridX = [], gridY = [], gridZ = [];
    for (let j = 0; j < n; j++) {
      const rx = [], ry = [], rz = [];
      for (let i = 0; i < n; i++) {
        rx.push(-1.5 + (3 * i) / (n - 1));
        ry.push(-1.5 + (3 * j) / (n - 1));
        rz.push(outputData[j * n + i]);
      }
      gridX.push(rx); gridY.push(ry); gridZ.push(rz);
    }
    return { gridX, gridY, gridZ };
  });
}

// ─── Per-neuron activation map (20×20) ───────────────────────────────────────
const MINI_N = 20;

export function computeNeuronGrid(model, layerIndex, neuronIndex) {
  // Build sub-model (functional API, using the sequential layer at layerIndex)
  const targetLayer = model.layers[layerIndex];
  const subModel = tf.model({ inputs: model.inputs, outputs: targetLayer.output });

  const n = MINI_N;
  const xs = [];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      xs.push([-1.5 + (3 * i) / (n - 1), -1.5 + (3 * j) / (n - 1)]);
    }
  }

  const inputT = tf.tensor2d(xs);
  const outputT = subModel.predict(inputT);
  const flat = Array.from(outputT.dataSync());
  const numNeurons = flat.length / (n * n);

  tf.dispose([inputT, outputT]);
  // Note: subModel shares weights with model, do NOT dispose its layers

  const grid = [];
  for (let j = 0; j < n; j++) {
    const row = [];
    for (let i = 0; i < n; i++) {
      row.push(flat[(j * n + i) * numNeurons + neuronIndex] ?? 0);
    }
    grid.push(row);
  }
  return grid;
}

// ─── Weight extraction ────────────────────────────────────────────────────────
export function extractWeights(model) {
  return model.layers
    .filter(l => l.getWeights().length >= 2)
    .map((l, idx) => {
      const [w] = l.getWeights();
      return { layerIndex: idx, weights: w.arraySync() };
    });
}
