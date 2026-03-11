/**
 * LeftPanel.jsx – Dataset selection, noise/ratio controls, hyperparameter sliders.
 */
import { useNN } from '../store/useNNStore';
import { DATASETS } from '../utils/datasets';
import './LeftPanel.css';

const ACTIVATION_OPTIONS = ['relu', 'sigmoid', 'tanh', 'linear'];
const REG_OPTIONS = ['none', 'l1', 'l2'];
const LR_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];

export default function LeftPanel({ onRegenerateData, onRebuildModel }) {
  const { state, dispatch } = useNN();

  const set = (key, val) => dispatch({ type: 'SET', payload: { [key]: val } });

  return (
    <aside className="left-panel">
      {/* ── Dataset ── */}
      <section className="panel-section">
        <h3 className="section-title">Dataset</h3>
        <div className="dataset-grid">
          {Object.keys(DATASETS).map(key => (
            <button
              key={key}
              className={`dataset-btn ${state.datasetType === key ? 'active' : ''}`}
              onClick={() => {
                set('datasetType', key);
                onRegenerateData(key, state.noise, state.trainRatio);
              }}
            >
              {key.charAt(0).toUpperCase() + key.slice(1)}
            </button>
          ))}
        </div>

        <label className="slider-label">
          <span>Noise <em>{state.noise.toFixed(2)}</em></span>
          <input type="range" min="0" max="1" step="0.01"
            value={state.noise}
            onChange={e => {
              const v = parseFloat(e.target.value);
              set('noise', v);
              onRegenerateData(state.datasetType, v, state.trainRatio);
            }}
          />
        </label>

        <label className="slider-label">
          <span>Train ratio <em>{Math.round(state.trainRatio * 100)}%</em></span>
          <input type="range" min="0.5" max="0.9" step="0.05"
            value={state.trainRatio}
            onChange={e => {
              const v = parseFloat(e.target.value);
              set('trainRatio', v);
              onRegenerateData(state.datasetType, state.noise, v);
            }}
          />
        </label>
      </section>

      {/* ── Architecture ── */}
      <section className="panel-section">
        <h3 className="section-title">Architecture</h3>

        <div className="row-control">
          <span>Hidden Layers</span>
          <div className="btn-group">
            <button onClick={() => { dispatch({ type: 'REMOVE_LAYER' }); onRebuildModel(); }}>−</button>
            <span className="count-badge">{state.hiddenLayers.length}</span>
            <button onClick={() => { dispatch({ type: 'ADD_LAYER' }); onRebuildModel(); }}>+</button>
          </div>
        </div>

        {state.hiddenLayers.map(([units, act], i) => (
          <div key={i} className="layer-row">
            <span className="layer-label">L{i + 1}</span>
            <div className="btn-group small">
              <button onClick={() => { dispatch({ type: 'SET_NEURONS', idx: i, units: units - 1 }); onRebuildModel(); }}>−</button>
              <span className="count-badge">{units}</span>
              <button onClick={() => { dispatch({ type: 'SET_NEURONS', idx: i, units: units + 1 }); onRebuildModel(); }}>+</button>
            </div>
            <select value={act} onChange={e => {
              dispatch({ type: 'SET_HIDDEN_LAYER', idx: i, act: e.target.value });
              onRebuildModel();
            }}>
              {ACTIVATION_OPTIONS.map(a => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>
        ))}
      </section>

      {/* ── Hyperparameters ── */}
      <section className="panel-section">
        <h3 className="section-title">Hyperparameters</h3>

        <label className="control-label">
          <span>Learning Rate</span>
          <select value={state.learningRate}
            onChange={e => { set('learningRate', parseFloat(e.target.value)); onRebuildModel(); }}>
            {LR_VALUES.map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>

        <label className="control-label">
          <span>Regularization</span>
          <select value={state.regularization}
            onChange={e => { set('regularization', e.target.value); onRebuildModel(); }}>
            {REG_OPTIONS.map(r => <option key={r} value={r}>{r.toUpperCase()}</option>)}
          </select>
        </label>

        {state.regularization !== 'none' && (
          <label className="slider-label">
            <span>Reg. Rate <em>{state.regRate.toFixed(4)}</em></span>
            <input type="range" min="0.0001" max="0.1" step="0.0001"
              value={state.regRate}
              onChange={e => { set('regRate', parseFloat(e.target.value)); onRebuildModel(); }}
            />
          </label>
        )}

        <label className="slider-label">
          <span>Batch Size <em>{state.batchSize}</em></span>
          <input type="range" min="1" max="256" step="1"
            value={state.batchSize}
            onChange={e => set('batchSize', parseInt(e.target.value))}
          />
        </label>
      </section>
    </aside>
  );
}
