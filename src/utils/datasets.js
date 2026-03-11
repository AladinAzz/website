/**
 * datasets.js – Toy dataset generators.
 * Each generator returns { trainX, trainY, testX, testY }
 * where X is an array of [x1, x2] pairs and Y is an array of 0 or 1.
 */

// Gaussian noise helper
function gauss(mean, std) {
  // Box-Muller transform
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function addNoise(val, amount) {
  return val + gauss(0, amount * 0.5);
}

function splitData(points, trainRatio) {
  const shuffled = [...points].sort(() => Math.random() - 0.5);
  const split = Math.floor(shuffled.length * trainRatio);
  const train = shuffled.slice(0, split);
  const test = shuffled.slice(split);
  return {
    trainX: train.map(p => [p[0], p[1]]),
    trainY: train.map(p => p[2]),
    testX: test.map(p => [p[0], p[1]]),
    testY: test.map(p => p[2]),
  };
}

// ─── Circle ───────────────────────────────────────────────────────────────────
export function generateCircle(n = 400, noise = 0, trainRatio = 0.8) {
  const points = [];
  for (let i = 0; i < n; i++) {
    const r = Math.random() < 0.5
      ? gauss(0.4, 0.15)   // inner cluster → label 0
      : gauss(0.85, 0.1);  // outer ring   → label 1
    const angle = Math.random() * 2 * Math.PI;
    const x = addNoise(r * Math.cos(angle), noise);
    const y = addNoise(r * Math.sin(angle), noise);
    const label = r < 0.65 ? 0 : 1;
    points.push([x * 2, y * 2, label]);
  }
  return splitData(points, trainRatio);
}

// ─── XOR / Checkerboard ───────────────────────────────────────────────────────
export function generateXOR(n = 400, noise = 0, trainRatio = 0.8) {
  const points = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 2 - 1;
    const y = Math.random() * 2 - 1;
    const label = (x * y > 0) ? 1 : 0;
    points.push([addNoise(x, noise), addNoise(y, noise), label]);
  }
  return splitData(points, trainRatio);
}

// ─── Two Spirals ──────────────────────────────────────────────────────────────
export function generateSpirals(n = 400, noise = 0, trainRatio = 0.8) {
  const points = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const angle = (i / half) * 3 * Math.PI;
    const r = (i / half) * 0.9 + 0.1;
    const x = addNoise(r * Math.cos(angle), noise * 0.2);
    const y = addNoise(r * Math.sin(angle), noise * 0.2);
    points.push([x, y, 0]);
    points.push([-x, -y, 1]);
  }
  return splitData(points, trainRatio);
}

// ─── Gaussian Clusters ────────────────────────────────────────────────────────
export function generateGaussian(n = 400, noise = 0, trainRatio = 0.8) {
  const points = [];
  const centers = [
    [-0.5, -0.5, 0],
    [0.5, 0.5, 1],
  ];
  for (let i = 0; i < n; i++) {
    const c = centers[i % 2];
    const spread = 0.25 + noise * 0.3;
    points.push([gauss(c[0], spread), gauss(c[1], spread), c[2]]);
  }
  return splitData(points, trainRatio);
}

export const DATASETS = {
  circle: generateCircle,
  xor: generateXOR,
  spirals: generateSpirals,
  gaussian: generateGaussian,
};
