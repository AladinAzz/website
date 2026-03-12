/**
 * App.jsx – Root component with CUDA server-backed training loop.
 * All heavy computation (model building, training, grid inference) runs on
 * the GPU server via HTTP. Falls back to browser-side TF.js if unreachable.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { NNProvider, useNN } from './store/useNNStore';
import { DATASETS } from './utils/datasets';
import { apiBuildModel, apiTrain, apiGetGrid, apiGetNeuronGrid, apiGetWeights, checkServer } from './utils/api';
import { buildModel, trainEpoch, computeGrid, computeNeuronGrid, extractWeights } from './utils/model';
import LeftPanel from './components/LeftPanel';
import NetworkGraph from './components/NetworkGraph';
import OutputCanvas from './components/OutputCanvas';
import TopBar from './components/TopBar';
import './App.css';

function AppInner() {
  const { state, dispatch } = useNN();
  const stateRef = useRef(state);
  const modelRef = useRef(null);       // browser-side model (fallback only)
  const animFrameRef = useRef(null);
  const trainingRef = useRef(false);
  const epochCountRef = useRef(0);
  const [serverMode, setServerMode] = useState(null); // null = checking, true = CUDA, false = browser

  // Keep stateRef in sync so callbacks always see current state
  useEffect(() => { stateRef.current = state; }, [state]);

  // ── Check if CUDA server is reachable on mount ─────────────────────────────
  useEffect(() => {
    checkServer().then(status => {
      setServerMode(status.connected);
      if (status.connected) {
        console.log(`🚀 CUDA server connected: ${status.backend} (${status.gpu})`);
      } else {
        console.log('⚠️ CUDA server not reachable — using browser TF.js');
      }
    });
  }, []);

  // ── Generate data (client-side, lightweight) ───────────────────────────────
  const generateData = useCallback((type, noise, ratio) => {
    const gen = DATASETS[type] || DATASETS.circle;
    const data = gen(400, noise, ratio);
    dispatch({ type: 'SET', payload: { data, datasetType: type, noise, trainRatio: ratio } });
  }, [dispatch]);

  // ── Build model ────────────────────────────────────────────────────────────
  const buildAndSetModel = useCallback(async () => {
    const { hiddenLayers, learningRate, regularization, regRate } = stateRef.current;
    epochCountRef.current = 0;
    dispatch({ type: 'SET', payload: { epoch: 0, trainLoss: null, testLoss: null, gridData: null, neuronGrids: {}, weightData: [] } });

    if (serverMode) {
      // Build on CUDA server
      try {
        await apiBuildModel(hiddenLayers, learningRate, regularization, regRate);
        console.log('✅ Model built on CUDA server');
      } catch (err) {
        console.error('Server build failed:', err);
      }
    } else {
      // Fallback: build in browser
      if (modelRef.current) { try { modelRef.current.dispose(); } catch (_) {} }
      modelRef.current = buildModel(hiddenLayers, learningRate, regularization, regRate);
    }
  }, [dispatch, serverMode]);

  // ── Visualization update ───────────────────────────────────────────────────
  const updateVisualizations = useCallback(async () => {
    const { hiddenLayers } = stateRef.current;

    if (serverMode) {
      try {
        const [gridData, { weights: weightData }] = await Promise.all([
          apiGetGrid(),
          apiGetWeights(),
        ]);

        const neuronGrids = {};
        const neuronPromises = [];
        hiddenLayers.forEach((_, li) => {
          const [units] = hiddenLayers[li];
          for (let ni = 0; ni < units; ni++) {
            const key = `L${li + 1}N${ni}`;
            neuronPromises.push(
              apiGetNeuronGrid(li, ni).then(r => { neuronGrids[key] = r.grid; }).catch(() => {})
            );
          }
        });
        await Promise.all(neuronPromises);

        dispatch({ type: 'SET', payload: { gridData, weightData, neuronGrids } });
      } catch (err) {
        console.error('Visualization error:', err);
      }
    } else {
      // Browser fallback
      const model = modelRef.current;
      if (!model) return;
      const gridData = computeGrid(model);
      const weightData = extractWeights(model);
      const neuronGrids = {};
      hiddenLayers.forEach((_, li) => {
        const [units] = hiddenLayers[li];
        for (let ni = 0; ni < units; ni++) {
          const key = `L${li + 1}N${ni}`;
          try { neuronGrids[key] = computeNeuronGrid(model, li, ni); } catch (_) {}
        }
      });
      dispatch({ type: 'SET', payload: { gridData, weightData, neuronGrids } });
    }
  }, [dispatch, serverMode]);

  // ── Training loop ──────────────────────────────────────────────────────────
  const GRID_INTERVAL = 3;
  const EPOCHS_PER_BATCH = 5; // CUDA: train 5 epochs per server call for throughput

  const runEpoch = useCallback(async () => {
    if (!trainingRef.current) return;
    const { data, batchSize } = stateRef.current;
    if (!data) {
      animFrameRef.current = requestAnimationFrame(runEpoch);
      return;
    }

    try {
      if (serverMode) {
        // ── CUDA server: batch N epochs per request ──
        const result = await apiTrain(
          data.trainX, data.trainY,
          data.testX, data.testY,
          batchSize,
          EPOCHS_PER_BATCH
        );
        // Update UI with the last epoch's losses
        const lastLoss = result.losses[result.losses.length - 1];
        epochCountRef.current += result.epochs;
        dispatch({ type: 'SET', payload: {
          epoch: epochCountRef.current,
          trainLoss: lastLoss.trainLoss,
          testLoss: lastLoss.testLoss,
        }});
        if (epochCountRef.current % GRID_INTERVAL < EPOCHS_PER_BATCH) {
          await updateVisualizations();
        }
      } else {
        // ── Browser fallback: 1 epoch at a time ──
        if (!modelRef.current) {
          animFrameRef.current = requestAnimationFrame(runEpoch);
          return;
        }
        const { trainLoss, testLoss } = await trainEpoch(
          modelRef.current,
          data.trainX, data.trainY,
          data.testX, data.testY,
          batchSize
        );
        epochCountRef.current += 1;
        dispatch({ type: 'SET', payload: { epoch: epochCountRef.current, trainLoss, testLoss } });
        if (epochCountRef.current % GRID_INTERVAL === 0) updateVisualizations();
      }
    } catch (e) {
      console.error('Training error:', e);
    }

    if (trainingRef.current) {
      animFrameRef.current = requestAnimationFrame(runEpoch);
    }
  }, [dispatch, updateVisualizations, serverMode]);

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
    if (!data) return;
    try {
      if (serverMode) {
        const result = await apiTrain(
          data.trainX, data.trainY,
          data.testX, data.testY,
          batchSize,
          1  // single epoch for step
        );
        const lastLoss = result.losses[0];
        epochCountRef.current += 1;
        dispatch({ type: 'SET', payload: {
          epoch: epochCountRef.current,
          trainLoss: lastLoss.trainLoss,
          testLoss: lastLoss.testLoss,
        }});
      } else {
        if (!modelRef.current) return;
        const { trainLoss, testLoss } = await trainEpoch(
          modelRef.current,
          data.trainX, data.trainY,
          data.testX, data.testY,
          batchSize
        );
        epochCountRef.current += 1;
        dispatch({ type: 'SET', payload: { epoch: epochCountRef.current, trainLoss, testLoss } });
      }
      await updateVisualizations();
    } catch (e) {
      console.error('Step error:', e);
    }
  }, [dispatch, updateVisualizations, serverMode]);

  const handleReset = useCallback(() => {
    handlePause();
    const { datasetType, noise, trainRatio } = stateRef.current;
    generateData(datasetType, noise, trainRatio);
    setTimeout(buildAndSetModel, 50);
  }, [handlePause, generateData, buildAndSetModel]);

  // ── Initialize once on mount ──────────────────────────────────────────────
  useEffect(() => {
    generateData('circle', 0, 0.8);
    // Wait for server mode detection before building model
    const timer = setTimeout(() => buildAndSetModel(), 200);
    return () => {
      clearTimeout(timer);
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
        serverMode={serverMode}
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
