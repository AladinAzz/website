/**
 * useNNStore.js – Global application state using React context + useReducer.
 */
import { createContext, useContext, useReducer } from 'react';

export const initialState = {
  // ── Dataset ──────────────────────────────────────────────────────────────
  datasetType: 'circle',      // 'circle' | 'xor' | 'spirals' | 'gaussian'
  noise: 0.0,
  trainRatio: 0.8,
  data: null,                 // { trainX, trainY, testX, testY }

  // ── Architecture ─────────────────────────────────────────────────────────
  // hiddenLayers: array of [neuronCount, activationName]
  hiddenLayers: [[4, 'relu'], [2, 'relu']],
  globalActivation: 'relu',

  // ── Hyperparameters ───────────────────────────────────────────────────────
  learningRate: 0.01,
  regularization: 'none',     // 'none' | 'l1' | 'l2'
  regRate: 0.001,
  batchSize: 32,

  // ── Training state ────────────────────────────────────────────────────────
  isTraining: false,
  epoch: 0,
  trainLoss: null,
  testLoss: null,

  // ── Visualization ─────────────────────────────────────────────────────────
  plotMode: '2d',             // '2d' | '3d'
  gridData: null,             // { gridX, gridY, gridZ }
  weightData: [],             // from extractWeights()
  neuronGrids: {},            // { 'L2N0': grid2dArray, ... }
};

function reducer(state, action) {
  switch (action.type) {
    case 'SET': return { ...state, ...action.payload };
    case 'SET_HIDDEN_LAYER':
      return {
        ...state,
        hiddenLayers: state.hiddenLayers.map((l, i) =>
          i === action.idx ? [action.units ?? l[0], action.act ?? l[1]] : l
        ),
      };
    case 'ADD_LAYER':
      return {
        ...state,
        hiddenLayers: [...state.hiddenLayers, [4, state.globalActivation]],
      };
    case 'REMOVE_LAYER':
      if (state.hiddenLayers.length <= 0) return state;
      return {
        ...state,
        hiddenLayers: state.hiddenLayers.slice(0, -1),
      };
    case 'SET_NEURONS':
      return {
        ...state,
        hiddenLayers: state.hiddenLayers.map((l, i) =>
          i === action.idx ? [Math.max(1, action.units), l[1]] : l
        ),
      };
    default: return state;
  }
}

const NNContext = createContext(null);

export function NNProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <NNContext.Provider value={{ state, dispatch }}>
      {children}
    </NNContext.Provider>
  );
}

export function useNN() {
  return useContext(NNContext);
}
