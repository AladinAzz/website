/**
 * api.js – HTTP client for the CUDA training server.
 * Falls back to browser-side TensorFlow.js if the server is unreachable.
 */

const BASE = '/api';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Server error ${res.status}`);
  }
  return res.json();
}

// ── Server health check ───────────────────────────────────────────────────────
export async function checkServer() {
  try {
    const data = await request('/status');
    return { connected: true, ...data };
  } catch (_) {
    return { connected: false, backend: null, isCuda: false };
  }
}

// ── Build model on GPU ────────────────────────────────────────────────────────
export async function apiBuildModel(hiddenLayers, learningRate, regularization, regRate) {
  return request('/build', {
    method: 'POST',
    body: JSON.stringify({ hiddenLayers, learningRate, regularization, regRate }),
  });
}

// ── Train N epochs on CUDA ────────────────────────────────────────────────────
export async function apiTrain(trainX, trainY, testX, testY, batchSize, epochs = 5) {
  return request('/train', {
    method: 'POST',
    body: JSON.stringify({ trainX, trainY, testX, testY, batchSize, epochs }),
  });
}

// ── Get decision boundary grid (50×50) ────────────────────────────────────────
export async function apiGetGrid() {
  return request('/grid');
}

// ── Get per-neuron activation grid ────────────────────────────────────────────
export async function apiGetNeuronGrid(layer, neuron) {
  return request(`/neuron-grid?layer=${layer}&neuron=${neuron}`);
}

// ── Get weight matrices ───────────────────────────────────────────────────────
export async function apiGetWeights() {
  return request('/weights');
}
