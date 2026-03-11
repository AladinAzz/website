/**
 * OutputCanvas.jsx – Plotly.js interactive 2D/3D visualization of the model's
 * decision boundary (surface/contour) with real data points and floor projections.
 */
import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import { useNN } from '../store/useNNStore';
import './OutputCanvas.css';

const COLORSCALE = [
  [0, '#f97316'],    // class 0 → orange
  [0.5, '#1e293b'],  // boundary  → dark
  [1, '#3b82f6'],    // class 1 → blue
];

export default function OutputCanvas() {
  const plotRef = useRef(null);
  const initializedRef = useRef(false);
  const { state } = useNN();
  const { gridData, data, plotMode } = state;

  const trainPoints0 = data ? data.trainX.filter((_, i) => data.trainY[i] === 0) : [];
  const trainPoints1 = data ? data.trainX.filter((_, i) => data.trainY[i] === 1) : [];
  const testPoints0  = data ? data.testX.filter((_, i) => data.testY[i] === 0) : [];
  const testPoints1  = data ? data.testX.filter((_, i) => data.testY[i] === 1) : [];

  useEffect(() => {
    if (!plotRef.current) return;
    initializedRef.current = false;
  }, [plotMode]);

  useEffect(() => {
    if (!plotRef.current || !data) return;

    const is3d = plotMode === '3d';

    // ── Build traces ────────────────────────────────────────────────────────
    const traces = [];

    if (gridData) {
      if (is3d) {
        // 3D surface
        traces.push({
          type: 'surface',
          x: gridData.gridX[0],
          y: gridData.gridY.map(row => row[0]),
          z: gridData.gridZ,
          colorscale: COLORSCALE,
          opacity: 0.82,
          showscale: false,
          name: 'Decision Surface',
          contours: {
            z: {
              show: true,
              usecolormap: true,
              highlightcolor: '#fff',
              project: { z: true },
            },
          },
        });

        // Floor contour projection
        traces.push({
          type: 'contour',
          x: gridData.gridX[0],
          y: gridData.gridY.map(row => row[0]),
          z: gridData.gridZ,
          colorscale: COLORSCALE,
          showscale: false,
          name: 'Floor projection',
          // Plotly hack: use surface_color with a flat z to project to floor
        });
      } else {
        // 2D filled contour
        traces.push({
          type: 'contour',
          x: gridData.gridX[0],
          y: gridData.gridY.map(r => r[0]),
          z: gridData.gridZ,
          colorscale: COLORSCALE,
          showscale: false,
          opacity: 0.85,
          contours: {
            coloring: 'fill',
            showlines: true,
            start: 0.5,
            end: 0.5,
            size: 0,
          },
          name: 'Boundary',
          line: { width: 2 },
        });
      }
    }

    // Data points
    if (is3d) {
      const addScatter3d = (pts, label, color, symbol) => {
        if (!pts.length) return;
        traces.push({
          type: 'scatter3d',
          mode: 'markers',
          x: pts.map(p => p[0]),
          y: pts.map(p => p[1]),
          z: pts.map(p => 0.5),
          marker: { color, size: 4, symbol: symbol || 'circle', opacity: 0.9 },
          name: label,
        });
      };
      addScatter3d(trainPoints0, 'Train 0', '#f97316', 'circle');
      addScatter3d(trainPoints1, 'Train 1', '#3b82f6', 'circle');
      addScatter3d(testPoints0, 'Test 0', '#fb923c', 'diamond');
      addScatter3d(testPoints1, 'Test 1', '#60a5fa', 'diamond');
    } else {
      const addScatter = (pts, label, color, sym) => {
        if (!pts.length) return;
        traces.push({
          type: 'scatter',
          mode: 'markers',
          x: pts.map(p => p[0]),
          y: pts.map(p => p[1]),
          marker: { color, size: 7, symbol: sym || 'circle', line: { color: '#fff', width: 0.5 } },
          name: label,
        });
      };
      addScatter(trainPoints0, 'Train 0', '#f97316');
      addScatter(trainPoints1, 'Train 1', '#3b82f6');
      addScatter(testPoints0, 'Test 0', '#fb923c', 'diamond');
      addScatter(testPoints1, 'Test 1', '#60a5fa', 'diamond');
    }

    // ── Layout ───────────────────────────────────────────────────────────────
    const axis = {
      gridcolor: '#334155',
      zerolinecolor: '#475569',
      tickfont: { color: '#94a3b8' },
      titlefont: { color: '#94a3b8' },
    };

    const layout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#0f172a',
      margin: { l: 10, r: 10, t: 10, b: 10 },
      showlegend: false,
      font: { color: '#94a3b8' },
      ...(is3d ? {
        scene: {
          xaxis: { ...axis, range: [-1.5, 1.5], title: 'x₁', backgroundcolor: '#0f172a' },
          yaxis: { ...axis, range: [-1.5, 1.5], title: 'x₂', backgroundcolor: '#0f172a' },
          zaxis: { ...axis, range: [0, 1], title: 'P(y=1)', backgroundcolor: '#0f172a' },
          camera: { eye: { x: 1.4, y: -1.4, z: 1.0 } },
          bgcolor: '#0f172a',
        },
      } : {
        xaxis: { ...axis, range: [-1.5, 1.5], title: 'x₁' },
        yaxis: { ...axis, range: [-1.5, 1.5], title: 'x₂' },
      }),
    };

    const config = {
      responsive: true,
      displaylogo: false,
      displayModeBar: true,
      modeBarButtonsToRemove: ['resetCameraLastSave3d'],
    };

    if (!initializedRef.current) {
      Plotly.newPlot(plotRef.current, traces, layout, config);
      initializedRef.current = true;
    } else {
      Plotly.react(plotRef.current, traces, layout, config);
    }
  }, [gridData, data, plotMode]);

  return (
    <div className="output-canvas-wrapper">
      <div ref={plotRef} className="plotly-chart" />
    </div>
  );
}
