import { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import type { DashboardData } from '../../types';
import { ChartContainer } from '../shared/ChartContainer';
import { Tooltip } from '../shared/Tooltip';

interface Props {
  data: DashboardData;
  activeModels: Set<string>;
  selectedThreshold: number;
}

export function SaturationDiagnostic({ data, activeModels, selectedThreshold }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tip, setTip] = useState({ x: 0, y: 0, visible: false, content: '' });

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth || 500;
    const height = svgRef.current.clientHeight || 320;
    const margin = { top: 10, right: 20, bottom: 35, left: 50 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const thresholds = data.meta.thresholds;
    const x = d3.scaleLinear().domain([0.6, 1.0]).range([0, w]);
    const y = d3.scaleLinear().domain([0, 105]).range([h, 0]);

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
      .call(d3.axisBottom(x).tickValues(thresholds).tickFormat(d3.format('.1f')))
      .selectAll('text,line,path').attr('stroke', '#64748b').attr('fill', '#94a3b8');

    g.select<SVGGElement>('.y-axis')
      .call(d3.axisLeft(y).ticks(5).tickFormat(d => `${d}%`))
      .selectAll('text,line,path').attr('stroke', '#64748b').attr('fill', '#94a3b8');

    const line = d3.line<{ t: number; pct: number }>()
      .x(d => x(d.t)).y(d => y(d.pct))
      .curve(d3.curveMonotoneX);

    const models = Array.from(activeModels);

    // Area fill
    const area = d3.area<{ t: number; pct: number }>()
      .x(d => x(d.t)).y0(h).y1(d => y(d.pct))
      .curve(d3.curveMonotoneX);

    const areas = g.selectAll<SVGPathElement, string>('.model-area').data(models, d => d);
    areas.enter()
      .append('path').attr('class', 'model-area')
      .attr('opacity', 0)
      .merge(areas)
      .attr('fill', m => data.meta.modelColors[m])
      .transition().duration(400).ease(d3.easeCubicOut)
      .attr('opacity', 0.1)
      .attr('d', m => {
        const pts = thresholds.map(t => ({ t, pct: data.saturation[m]?.[String(t)] ?? 0 }));
        return area(pts);
      });
    areas.exit().transition().duration(300).attr('opacity', 0).remove();

    const paths = g.selectAll<SVGPathElement, string>('.model-line').data(models, d => d);
    paths.enter()
      .append('path').attr('class', 'model-line')
      .attr('fill', 'none').attr('stroke-width', 2.5).attr('opacity', 0)
      .merge(paths)
      .attr('stroke', m => data.meta.modelColors[m])
      .transition().duration(400).ease(d3.easeCubicOut)
      .attr('opacity', 1)
      .attr('d', m => {
        const pts = thresholds.map(t => ({ t, pct: data.saturation[m]?.[String(t)] ?? 0 }));
        return line(pts);
      });
    paths.exit().transition().duration(300).attr('opacity', 0).remove();

    // Dots
    const dotData = models.flatMap(m =>
      thresholds.map(t => ({
        key: `${m}-${t}`, model: m, threshold: t,
        pct: data.saturation[m]?.[String(t)] ?? 0,
      }))
    );
    const dots = g.selectAll<SVGCircleElement, typeof dotData[0]>('.dot').data(dotData, d => d.key);
    dots.enter()
      .append('circle').attr('class', 'dot').attr('r', 4).attr('opacity', 0)
      .on('mouseover', (event, d) => {
        setTip({ x: event.clientX, y: event.clientY, visible: true,
          content: `${d.model}\nThreshold: ${d.threshold}\n>= threshold: ${d.pct.toFixed(2)}%` });
      })
      .on('mouseout', () => setTip(t => ({ ...t, visible: false })))
      .merge(dots)
      .attr('fill', d => data.meta.modelColors[d.model])
      .transition().duration(400).ease(d3.easeCubicOut)
      .attr('cx', d => x(d.threshold)).attr('cy', d => y(d.pct))
      .attr('opacity', 1);
    dots.exit().transition().duration(300).attr('opacity', 0).remove();

    // Vertical reference line at selected threshold
    g.selectAll('.ref-line').remove();
    g.selectAll('.ref-label').remove();
    const refX = x(selectedThreshold);
    g.append('line').attr('class', 'ref-line')
      .attr('x1', refX).attr('x2', refX)
      .attr('y1', 0).attr('y2', h)
      .attr('stroke', '#f97316').attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '6,4');
    g.append('text').attr('class', 'ref-label')
      .attr('x', refX + 4).attr('y', 14)
      .attr('fill', '#f97316').attr('font-size', '11px')
      .text(`t=${selectedThreshold}`);
  });

  return (
    <>
      <ChartContainer title="Saturation: % Patient Pairs >= Threshold">
        {({ width, height }) => (
          <svg ref={svgRef} width={width} height={height} />
        )}
      </ChartContainer>
      <Tooltip {...tip} />
    </>
  );
}
