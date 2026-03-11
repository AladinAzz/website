/**
 * App.jsx – Root component with stable training loop using refs to avoid
 * dependency loops in useEffect. All mutable training state lives in refs.
 */
import { useCallback, useEffect, useRef } from 'react';
import { NNProvider, useNN } from './store/useNNStore';
import { DATASETS } from './utils/datasets';
import { buildModel, trainEpoch, computeGrid, computeNeuronGrid, extractWeights } from './utils/model';
import LeftPanel from './components/LeftPanel';
import NetworkGraph from './components/NetworkGraph';
import OutputCanvas from './components/OutputCanvas';
import TopBar from './components/TopBar';
import './App.css';

function AppInner() {
  const { state, dispatch } = useNN();
  const stateRef = useRef(state);
  const modelRef = useRef(null);
  const animFrameRef = useRef(null);
  const trainingRef = useRef(false);
  const epochCountRef = useRef(0);

  // Keep stateRef in sync so callbacks always see current state
  useEffect(() => { stateRef.current = state; }, [state]);

  // ── Generate data ─────────────────────────────────────────────────────────
  const generateData = useCallback((type, noise, ratio) => {
    const gen = DATASETS[type] || DATASETS.circle;
    const data = gen(400, noise, ratio);
    dispatch({ type: 'SET', payload: { data, datasetType: type, noise, trainRatio: ratio } });
  }, [dispatch]);

  // ── Build model (uses stateRef to read current config) ────────────────────
  const buildAndSetModel = useCallback(() => {
    if (modelRef.current) { try { modelRef.current.dispose(); } catch (_) {} }
    const { hiddenLayers, learningRate, regularization, regRate } = stateRef.current;
    modelRef.current = buildModel(hiddenLayers, learningRate, regularization, regRate);
    epochCountRef.current = 0;
    dispatch({ type: 'SET', payload: { epoch: 0, trainLoss: null, testLoss: null, gridData: null, neuronGrids: {}, weightData: [] } });
  }, [dispatch]);

  // ── Visualization update ──────────────────────────────────────────────────
  const updateVisualizations = useCallback(() => {
    const model = modelRef.current;
    if (!model) return;
    const { hiddenLayers } = stateRef.current;

    const gridData = computeGrid(model);
    const weightData = extractWeights(model);

    const neuronGrids = {};
    hiddenLayers.forEach((_, li) => {
      const [units] = hiddenLayers[li];
      for (let ni = 0; ni < units; ni++) {
        const key = `L${li + 1}N${ni}`;
        try { neuronGrids[key] = computeNeuronGrid(model, li, ni); }
        catch (_) {}
      }
    });
    dispatch({ type: 'SET', payload: { gridData, weightData, neuronGrids } });
  }, [dispatch]);

  // ── Training loop using requestAnimationFrame ─────────────────────────────
  const GRID_INTERVAL = 3;
  const runEpoch = useCallback(async () => {
    if (!trainingRef.current) return;
    const { data, batchSize } = stateRef.current;
    if (!data || !modelRef.current) {
      animFrameRef.current = requestAnimationFrame(runEpoch);
      return;
    }
    try {
      const { trainLoss, testLoss } = await trainEpoch(
        modelRef.current,
        data.trainX, data.trainY,
        data.testX, data.testY,
        batchSize
      );
      epochCountRef.current += 1;
      dispatch({ type: 'SET', payload: { epoch: epochCountRef.current, trainLoss, testLoss } });
      if (epochCountRef.current % GRID_INTERVAL === 0) updateVisualizations();
    } catch (e) {
      console.error('Training error:', e);
    }
    if (trainingRef.current) {
      animFrameRef.current = requestAnimationFrame(runEpoch);
    }
  }, [dispatch, updateVisualizations]);

  // ── Controls ──────────────────────────────────────────────────────────────
  const handlePlay = useCallback(() => {
    if (trainingRef.current) return;
    trainingRef.current = true;
    dispatch({ type: 'SET', payload: { isTraining: true } });
    animFrameRef.current = requestAnimationFrame(runEpoch);
  }, [dispatch, runEpoch]);

  const handlePause = useCallback(() => {
    trainingRef.current = false;
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    dispatch({ type: 'SET', payload: { isTraining: false } });
  }, [dispatch]);

  const handleStep = useCallback(async () => {
    if (trainingRef.current) return;
    const { data, batchSize } = stateRef.current;
    if (!data || !modelRef.current) return;
    try {
      const { trainLoss, testLoss } = await trainEpoch(
        modelRef.current,
        data.trainX, data.trainY,
        data.testX, data.testY,
        batchSize
      );
      epochCountRef.current += 1;
      dispatch({ type: 'SET', payload: { epoch: epochCountRef.current, trainLoss, testLoss } });
      updateVisualizations();
    } catch (e) {
      console.error('Step error:', e);
    }
  }, [dispatch, updateVisualizations]);

  const handleReset = useCallback(() => {
    handlePause();
    const { datasetType, noise, trainRatio } = stateRef.current;
    generateData(datasetType, noise, trainRatio);
    setTimeout(buildAndSetModel, 50);
  }, [handlePause, generateData, buildAndSetModel]);

  // ── Initialize once on mount ──────────────────────────────────────────────
  useEffect(() => {
    generateData('circle', 0, 0.8);
    buildAndSetModel();
    return () => {
      trainingRef.current = false;
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (modelRef.current) { try { modelRef.current.dispose(); } catch (_) {} }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="app">
      <TopBar
        onPlay={handlePlay}
        onPause={handlePause}
        onStep={handleStep}
        onReset={handleReset}
      />
      <div className="main-layout">
        <LeftPanel
          onRegenerateData={generateData}
          onRebuildModel={buildAndSetModel}
        />
        <main className="center-panel">
          <div className="panel-header">Network Graph</div>
          <NetworkGraph />
        </main>
        <section className="right-panel">
          <div className="panel-header">Output Visualization</div>
          <OutputCanvas />
        </section>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <NNProvider>
      <AppInner />
    </NNProvider>
  );
}
