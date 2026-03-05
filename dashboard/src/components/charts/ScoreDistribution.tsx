import { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import type { DashboardData } from '../../types';
import { ChartContainer } from '../shared/ChartContainer';
import { Tooltip } from '../shared/Tooltip';

interface Props {
  data: DashboardData;
  activeModels: Set<string>;
}

export function ScoreDistribution({ data, activeModels }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tip, setTip] = useState({ x: 0, y: 0, visible: false, content: '' });
  const [dragThreshold, setDragThreshold] = useState(0.6);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth || 500;
    const height = svgRef.current.clientHeight || 320;
    const margin = { top: 10, right: 60, bottom: 35, left: 45 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const models = Array.from(activeModels);
    const x = d3.scaleBand<string>().domain(models).range([0, w]).padding(0.4);
    const y = d3.scaleLinear().domain([0.4, 1.0]).range([h, 0]);

    let g = svg.select<SVGGElement>('.chart-area');
    if (g.empty()) {
      g = svg.append('g').attr('class', 'chart-area')
        .attr('transform', `translate(${margin.left},${margin.top})`);
      g.append('g').attr('class', 'x-axis').attr('transform', `translate(0,${h})`);
      g.append('g').attr('class', 'y-axis');
    }

    svg.attr('width', width).attr('height', height);

    g.select<SVGGElement>('.x-axis')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x))
      .selectAll('text').attr('fill', '#94a3b8').attr('font-size', '10px');
    g.select<SVGGElement>('.x-axis').selectAll('line,path').attr('stroke', '#64748b');

    g.select<SVGGElement>('.y-axis')
      .call(d3.axisLeft(y).ticks(6).tickFormat(d3.format('.1f')))
      .selectAll('text,line,path').attr('stroke', '#64748b').attr('fill', '#94a3b8');

    // Box plots
    const dist = data.scoreDistribution.perPatientMax;
    const boxData = models.filter(m => dist[m]).map(m => ({ model: m, ...dist[m] }));

    // Whiskers (p5 to p95)
    const whiskers = g.selectAll<SVGLineElement, typeof boxData[0]>('.whisker')
      .data(boxData, d => d.model);
    whiskers.enter().append('line').attr('class', 'whisker')
      .merge(whiskers)
      .transition().duration(400)
      .attr('x1', d => x(d.model)! + x.bandwidth() / 2)
      .attr('x2', d => x(d.model)! + x.bandwidth() / 2)
      .attr('y1', d => y(d.p5)).attr('y2', d => y(d.p95))
      .attr('stroke', '#94a3b8').attr('stroke-width', 1.5);
    whiskers.exit().remove();

    // Whisker caps
    const capData = boxData.flatMap(d => [
      { key: `${d.model}-lo`, model: d.model, val: d.p5 },
      { key: `${d.model}-hi`, model: d.model, val: d.p95 },
    ]);
    const caps = g.selectAll<SVGLineElement, typeof capData[0]>('.cap').data(capData, d => d.key);
    caps.enter().append('line').attr('class', 'cap')
      .merge(caps)
      .transition().duration(400)
      .attr('x1', d => x(d.model)! + x.bandwidth() * 0.25)
      .attr('x2', d => x(d.model)! + x.bandwidth() * 0.75)
      .attr('y1', d => y(d.val)).attr('y2', d => y(d.val))
      .attr('stroke', '#94a3b8').attr('stroke-width', 1.5);
    caps.exit().remove();

    // Boxes (p25 to p75)
    const boxes = g.selectAll<SVGRectElement, typeof boxData[0]>('.box')
      .data(boxData, d => d.model);
    boxes.enter().append('rect').attr('class', 'box')
      .on('mouseover', (event, d) => {
        const pct = interpolatePct(d.thresholdPct, dragThreshold);
        setTip({ x: event.clientX, y: event.clientY, visible: true,
          content: `${d.model}\nMedian: ${d.median.toFixed(4)}\nMean: ${d.mean.toFixed(4)}\nStd: ${d.std.toFixed(4)}\n>= ${dragThreshold}: ${pct.toFixed(1)}%` });
      })
      .on('mouseout', () => setTip(t => ({ ...t, visible: false })))
      .merge(boxes)
      .transition().duration(400)
      .attr('x', d => x(d.model)!)
      .attr('width', x.bandwidth())
      .attr('y', d => y(d.p75))
      .attr('height', d => Math.max(0, y(d.p25) - y(d.p75)))
      .attr('fill', d => data.meta.modelColors[d.model])
      .attr('fill-opacity', 0.3)
      .attr('stroke', d => data.meta.modelColors[d.model])
      .attr('stroke-width', 2)
      .attr('rx', 3);
    boxes.exit().remove();

    // Median line
    const medians = g.selectAll<SVGLineElement, typeof boxData[0]>('.median')
      .data(boxData, d => d.model);
    medians.enter().append('line').attr('class', 'median')
      .merge(medians)
      .transition().duration(400)
      .attr('x1', d => x(d.model)!)
      .attr('x2', d => x(d.model)! + x.bandwidth())
      .attr('y1', d => y(d.median)).attr('y2', d => y(d.median))
      .attr('stroke', '#f1f5f9').attr('stroke-width', 2);
    medians.exit().remove();

    // Mean diamond
    const means = g.selectAll<SVGPolygonElement, typeof boxData[0]>('.mean-diamond')
      .data(boxData, d => d.model);
    means.enter().append('polygon').attr('class', 'mean-diamond')
      .merge(means)
      .transition().duration(400)
      .attr('points', d => {
        const cx = x(d.model)! + x.bandwidth() / 2;
        const cy = y(d.mean);
        const s = 5;
        return `${cx},${cy - s} ${cx + s},${cy} ${cx},${cy + s} ${cx - s},${cy}`;
      })
      .attr('fill', '#f1f5f9');
    means.exit().remove();

    // Draggable threshold line
    g.selectAll('.threshold-line').remove();
    g.selectAll('.threshold-label').remove();

    const threshY = y(Math.max(0.4, Math.min(1.0, dragThreshold)));
    g.append('line').attr('class', 'threshold-line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', threshY).attr('y2', threshY)
      .attr('stroke', '#f97316').attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '6,4')
      .style('cursor', 'ns-resize')
      .call(d3.drag<SVGLineElement, unknown>()
        .on('drag', (event) => {
          const newT = Math.round(y.invert(event.y) * 20) / 20;
          setDragThreshold(Math.max(0.4, Math.min(1.0, newT)));
        }) as any
      );

    // Pct labels on right
    g.selectAll('.pct-label').remove();
    boxData.forEach(d => {
      const pct = interpolatePct(d.thresholdPct, dragThreshold);
      g.append('text').attr('class', 'pct-label')
        .attr('x', x(d.model)! + x.bandwidth() / 2)
        .attr('y', y(Math.max(0.4, dragThreshold)) - 10)
        .attr('text-anchor', 'middle')
        .attr('fill', data.meta.modelColors[d.model])
        .attr('font-size', '11px')
        .attr('font-weight', 'bold')
        .text(`${pct.toFixed(1)}%`);
    });

  });

  return (
    <>
      <ChartContainer title="Per-Patient MAX Similarity Distribution">
        {({ width, height }) => (
          <svg ref={svgRef} width={width} height={height} />
        )}
      </ChartContainer>
      <Tooltip {...tip} />
    </>
  );
}

function interpolatePct(thresholdPct: Record<string, number>, t: number): number {
  const keys = Object.keys(thresholdPct).map(Number).sort((a, b) => a - b);
  if (t <= keys[0]) return thresholdPct[String(keys[0])] ?? 100;
  if (t >= keys[keys.length - 1]) return thresholdPct[String(keys[keys.length - 1])] ?? 0;
  for (let i = 0; i < keys.length - 1; i++) {
    if (t >= keys[i] && t <= keys[i + 1]) {
      const ratio = (t - keys[i]) / (keys[i + 1] - keys[i]);
      const lo = thresholdPct[String(keys[i])] ?? 100;
      const hi = thresholdPct[String(keys[i + 1])] ?? 0;
      return lo + ratio * (hi - lo);
    }
  }
  return 0;
}
