/**
 * TopBar.jsx – Training controls: Play/Pause, Step, Reset, and metrics display.
 */
import { useNN } from '../store/useNNStore';
import './TopBar.css';

export default function TopBar({ onPlay, onPause, onStep, onReset, onToggle3d }) {
  const { state, dispatch } = useNN();
  const { isTraining, epoch, trainLoss, testLoss, plotMode } = state;

  const fmtLoss = v => (v !== null && v !== undefined) ? v.toFixed(4) : '—';

  return (
    <header className="top-bar">
      <div className="brand">
        <span className="brand-icon">⬡</span>
        <span className="brand-name">NeuralViz</span>
      </div>

      <div className="training-controls">
        <button
          className={`ctrl-btn primary ${isTraining ? 'active' : ''}`}
          onClick={isTraining ? onPause : onPlay}
          title={isTraining ? 'Pause training' : 'Start training'}
        >
          {isTraining ? '⏸ Pause' : '▶ Play'}
        </button>

        <button
          className="ctrl-btn"
          onClick={onStep}
          disabled={isTraining}
          title="Train one epoch"
        >
          ⏭ Step
        </button>

        <button
          className="ctrl-btn danger"
          onClick={onReset}
          title="Reset model and training"
        >
          ↺ Reset
        </button>
      </div>

      <div className="metrics">
        <div className="metric-chip">
          <span className="metric-label">Epoch</span>
          <span className="metric-value">{epoch}</span>
        </div>
        <div className="metric-chip">
          <span className="metric-label">Train Loss</span>
          <span className="metric-value loss-train">{fmtLoss(trainLoss)}</span>
        </div>
        <div className="metric-chip">
          <span className="metric-label">Test Loss</span>
          <span className="metric-value loss-test">{fmtLoss(testLoss)}</span>
        </div>
      </div>

      <div className="view-toggle">
        <button
          className={`toggle-btn ${plotMode === '2d' ? 'active' : ''}`}
          onClick={() => { dispatch({ type: 'SET', payload: { plotMode: '2d' } }); }}
        >2D</button>
        <button
          className={`toggle-btn ${plotMode === '3d' ? 'active' : ''}`}
          onClick={() => { dispatch({ type: 'SET', payload: { plotMode: '3d' } }); }}
        >3D</button>
      </div>
    </header>
  );
}
