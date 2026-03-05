import { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import type { DashboardData } from '../../types';
import { ChartContainer } from '../shared/ChartContainer';
import { MethodSelector } from '../controls/MethodSelector';
import { Tooltip } from '../shared/Tooltip';

interface Props {
  data: DashboardData;
  activeModels: Set<string>;
  selectedMethod: string;
  onMethodChange: (m: string) => void;
}

export function F1VsThreshold({ data, activeModels, selectedMethod, onMethodChange }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tip, setTip] = useState({ x: 0, y: 0, visible: false, content: '' });

  const renderChart = (width: number, height: number) => {
    const svg = d3.select(svgRef.current);
    const margin = { top: 10, right: 20, bottom: 35, left: 45 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const thresholds = data.meta.thresholds;
    const x = d3.scaleLinear().domain([0.6, 1.0]).range([0, w]);
    const y = d3.scaleLinear().domain([0, 1.05]).range([h, 0]);

    let g = svg.select<SVGGElement>('.chart-area');
    if (g.empty()) {
      g = svg.append('g').attr('class', 'chart-area')
        .attr('transform', `translate(${margin.left},${margin.top})`);
      g.append('g').attr('class', 'x-axis').attr('transform', `translate(0,${h})`);
      g.append('g').attr('class', 'y-axis');
      g.append('g').attr('class', 'grid');
    }

    // Update dimensions
    svg.attr('width', width).attr('height', height);
    g.attr('transform', `translate(${margin.left},${margin.top})`);

    // Axes
    g.select<SVGGElement>('.x-axis')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).tickValues(thresholds).tickFormat(d3.format('.1f')))
      .selectAll('text,line,path').attr('stroke', '#64748b').attr('fill', '#94a3b8');

    g.select<SVGGElement>('.y-axis')
      .call(d3.axisLeft(y).ticks(5))
      .selectAll('text,line,path').attr('stroke', '#64748b').attr('fill', '#94a3b8');

    // Grid
    const gridLines = g.select('.grid').selectAll<SVGLineElement, number>('line')
      .data([0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    gridLines.enter().append('line').merge(gridLines)
      .attr('x1', 0).attr('x2', w)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', '#1e293b').attr('stroke-width', 1);

    const line = d3.line<{ t: number; f1: number }>()
      .x(d => x(d.t))
      .y(d => y(d.f1))
      .curve(d3.curveMonotoneX);

    const models = Array.from(activeModels);

    // Lines
    const paths = g.selectAll<SVGPathElement, string>('.model-line').data(models, d => d);
    paths.enter()
      .append('path').attr('class', 'model-line')
      .attr('fill', 'none').attr('stroke-width', 2.5)
      .attr('opacity', 0)
      .merge(paths)
      .attr('stroke', m => data.meta.modelColors[m])
      .transition().duration(400).ease(d3.easeCubicOut)
      .attr('opacity', 1)
      .attr('d', m => {
        const pts = thresholds.map(t => ({
          t, f1: data.performance[m]?.[selectedMethod]?.[String(t)]?.F1 ?? 0
        }));
        return line(pts);
      });
    paths.exit().transition().duration(300).attr('opacity', 0).remove();

    // Dots
    const dotData = models.flatMap(m =>
      thresholds.map(t => ({
        key: `${m}-${t}`,
        model: m,
        threshold: t,
        f1: data.performance[m]?.[selectedMethod]?.[String(t)]?.F1 ?? 0,
      }))
    );
    const dots = g.selectAll<SVGCircleElement, typeof dotData[0]>('.dot').data(dotData, d => d.key);
    dots.enter()
      .append('circle').attr('class', 'dot')
      .attr('r', 4).attr('opacity', 0)
      .on('mouseover', (event, d) => {
        setTip({ x: event.clientX, y: event.clientY, visible: true,
          content: `${d.model}\nThreshold: ${d.threshold}\nF1: ${d.f1.toFixed(4)}` });
      })
      .on('mouseout', () => setTip(t => ({ ...t, visible: false })))
      .merge(dots)
      .attr('fill', d => data.meta.modelColors[d.model])
      .transition().duration(400).ease(d3.easeCubicOut)
      .attr('cx', d => x(d.threshold))
      .attr('cy', d => y(d.f1))
      .attr('opacity', 1);
    dots.exit().transition().duration(300).attr('opacity', 0).remove();
  };

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = svgRef.current;
    renderChart(svg.clientWidth || 500, svg.clientHeight || 320);
  });

  return (
    <>
      <ChartContainer
        title="F1 vs Threshold"
        controls={
          <MethodSelector
            methods={data.meta.methods}
            selected={selectedMethod}
            onChange={onMethodChange}
          />
        }
      >
        {({ width, height }) => (
          <svg ref={svgRef} width={width} height={height} />
        )}
      </ChartContainer>
      <Tooltip {...tip} />
    </>
  );
}
