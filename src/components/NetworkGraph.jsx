/**
 * NetworkGraph.jsx – D3.js visualization of neural network layers, neurons,
 * connecting weights (thickness + color), and per-neuron mini activation maps.
 */
import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useNN } from '../store/useNNStore';
import './NetworkGraph.css';

const NODE_R = 26;
const COL_W  = 115;
const ROW_H  = 70;
const PAD_X  = 55;
const PAD_Y  = 50;

const weightColor = d3.scaleLinear()
  .domain([-1, 0, 1])
  .range(['#f97316', '#e2e8f0', '#3b82f6'])
  .clamp(true);

const neuronColorScale = d3.scaleSequential(d3.interpolateRdBu).domain([1, 0]);

function buildLayers(hiddenLayers) {
  return [
    { label: 'Input',  neurons: ['x₁', 'x₂'], isInput: true },
    ...hiddenLayers.map(([n, act], i) => ({
      label: `H${i + 1} (${act})`,
      neurons: Array.from({ length: n }, (_, j) => `L${i + 1}N${j}`),
    })),
    { label: 'Output', neurons: ['ŷ'], isOutput: true },
  ];
}

export default function NetworkGraph() {
  const svgRef  = useRef(null);
  const { state } = useNN();
  const { hiddenLayers, weightData, neuronGrids } = state;

  const layers = buildLayers(hiddenLayers);
  const maxNeurons = Math.max(...layers.map(l => l.neurons.length));
  const svgW = PAD_X * 2 + COL_W * (layers.length - 1);
  const svgH = PAD_Y * 2 + ROW_H * maxNeurons;

  // ── Compute node positions ──────────────────────────────────────────────────
  function getNodePos() {
    const pos = {};
    layers.forEach((layer, li) => {
      const x = PAD_X + li * COL_W;
      const count = layer.neurons.length;
      const totalH = (count - 1) * ROW_H;
      const startY = (svgH - totalH) / 2;
      layer.neurons.forEach((nid, ni) => {
        pos[nid] = { x, y: startY + ni * ROW_H };
      });
    });
    return pos;
  }

  // ── Draw graph skeleton (edges + node circles) when topology changes ────────
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const nodePos = getNodePos();

    // Background grid
    svg.append('defs').append('pattern')
      .attr('id', 'grid').attr('width', 20).attr('height', 20)
      .attr('patternUnits', 'userSpaceOnUse')
      .append('path').attr('d', 'M 20 0 L 0 0 0 20')
      .attr('fill', 'none').attr('stroke', '#1e293b').attr('stroke-width', 0.5);
    svg.append('rect').attr('width', svgW).attr('height', svgH).attr('fill', 'url(#grid)');

    // ── Edges ────────────────────────────────────────────────────────────────
    const edgeGroup = svg.append('g').attr('class', 'edges');
    layers.forEach((layer, li) => {
      if (li === 0) return;
      const prevLayer = layers[li - 1];
      const wInfo = weightData[li - 1];

      prevLayer.neurons.forEach((src, si) => {
        layer.neurons.forEach((dst, di) => {
          let w = 0;
          if (wInfo && wInfo.weights && wInfo.weights[si]) {
            w = wInfo.weights[si][di] ?? 0;
          }
          const thickness = Math.abs(w) * 4 + 0.5;
          const color = weightColor(Math.max(-1, Math.min(1, w)));
          const p1 = nodePos[src];
          const p2 = nodePos[dst];
          edgeGroup.append('line')
            .attr('x1', p1.x).attr('y1', p1.y)
            .attr('x2', p2.x).attr('y2', p2.y)
            .attr('stroke', color)
            .attr('stroke-width', thickness)
            .attr('stroke-opacity', 0.7);
        });
      });
    });

    // ── Layer labels ─────────────────────────────────────────────────────────
    layers.forEach((layer, li) => {
      const x = PAD_X + li * COL_W;
      svg.append('text')
        .attr('x', x).attr('y', 18)
        .attr('text-anchor', 'middle')
        .attr('fill', '#475569')
        .attr('font-size', 10)
        .attr('font-weight', '600')
        .attr('letter-spacing', '0.05em')
        .text(layer.label);
    });

    // ── Nodes ────────────────────────────────────────────────────────────────
    layers.forEach((layer) => {
      layer.neurons.forEach((nid) => {
        const { x, y } = nodePos[nid];
        const g = svg.append('g')
          .attr('transform', `translate(${x},${y})`)
          .attr('class', 'neuron-group');

        if (layer.isInput || layer.isOutput) {
          g.append('circle')
            .attr('r', NODE_R)
            .attr('fill', layer.isInput ? '#0f2a56' : '#3b1d72')
            .attr('stroke', layer.isInput ? '#3b82f6' : '#8b5cf6')
            .attr('stroke-width', 1.5);
          g.append('text')
            .attr('text-anchor', 'middle').attr('dy', '0.35em')
            .attr('fill', '#e2e8f0').attr('font-size', 13).attr('font-weight', '500')
            .text(nid);
        } else {
          // Outer ring
          g.append('circle')
            .attr('r', NODE_R + 2)
            .attr('fill', 'none')
            .attr('stroke', '#334155')
            .attr('stroke-width', 1.5);

          // foreignObject canvas for mini activation map
          g.append('foreignObject')
            .attr('x', -NODE_R).attr('y', -NODE_R)
            .attr('width', NODE_R * 2).attr('height', NODE_R * 2)
            .html(`<canvas id="nc-${nid}" width="${NODE_R*2}" height="${NODE_R*2}" style="border-radius:50%;display:block;"></canvas>`);
        }
      });
    });
  }, [hiddenLayers, weightData]); // eslint-disable-line

  // ── Paint activation maps after D3 renders ──────────────────────────────────
  useEffect(() => {
    if (!neuronGrids || Object.keys(neuronGrids).length === 0) return;

    // Use requestAnimationFrame to wait for DOM update after D3
    const raf = requestAnimationFrame(() => {
      layers.forEach((layer) => {
        if (layer.isInput || layer.isOutput) return;
        layer.neurons.forEach((nid) => {
          const grid = neuronGrids[nid];
          if (!grid) return;
          const canvas = document.getElementById(`nc-${nid}`);
          if (!canvas) return;
          const ctx = canvas.getContext('2d');
          const n = grid.length;
          const cw = (NODE_R * 2) / n;
          const ch = (NODE_R * 2) / n;
          grid.forEach((row, rj) => {
            row.forEach((val, ri) => {
              ctx.fillStyle = neuronColorScale(val);
              ctx.fillRect(ri * cw, rj * ch, cw, ch);
            });
          });
        });
      });
    });
    return () => cancelAnimationFrame(raf);
  }, [neuronGrids, hiddenLayers]); // eslint-disable-line

  return (
    <div className="network-graph-container">
      <svg
        ref={svgRef}
        width={svgW}
        height={svgH}
        className="network-svg"
      />
    </div>
  );
}
