import { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import type { DashboardData } from '../../types';
import { ChartContainer } from '../shared/ChartContainer';
import { ThresholdSlider } from '../controls/ThresholdSlider';
import { Tooltip } from '../shared/Tooltip';

interface Props {
  data: DashboardData;
  activeModels: Set<string>;
  selectedThreshold: number;
  onThresholdChange: (t: number) => void;
}

export function F1VsTopK({ data, activeModels, selectedThreshold, onThresholdChange }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tip, setTip] = useState({ x: 0, y: 0, visible: false, content: '' });

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth || 500;
    const height = svgRef.current.clientHeight || 320;
    const margin = { top: 10, right: 20, bottom: 45, left: 45 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const methods = data.meta.methods;
    const x = d3.scalePoint<string>().domain(methods).range([0, w]).padding(0.1);
    const y = d3.scaleLinear().domain([0, 1.05]).range([h, 0]);

    let g = svg.select<SVGGElement>('.chart-area');
    if (g.empty()) {
      g = svg.append('g').attr('class', 'chart-area')
        .attr('transform', `translate(${margin.left},${margin.top})`);
      g.append('g').attr('class', 'x-axis').attr('transform', `translate(0,${h})`);
      g.append('g').attr('class', 'y-axis');
      g.append('g').attr('class', 'grid');
    }

    svg.attr('width', width).attr('height', height);

    g.select<SVGGElement>('.x-axis')
      .attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x))
      .selectAll('text').attr('fill', '#94a3b8').attr('font-size', '10px')
      .attr('transform', 'rotate(-20)').attr('text-anchor', 'end');
    g.select<SVGGElement>('.x-axis').selectAll('line,path').attr('stroke', '#64748b');

    g.select<SVGGElement>('.y-axis')
      .call(d3.axisLeft(y).ticks(5))
      .selectAll('text,line,path').attr('stroke', '#64748b').attr('fill', '#94a3b8');

    const gridLines = g.select('.grid').selectAll<SVGLineElement, number>('line')
      .data([0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    gridLines.enter().append('line').merge(gridLines)
      .attr('x1', 0).attr('x2', w)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', '#1e293b');

    const line = d3.line<{ method: string; f1: number }>()
      .x(d => x(d.method)!)
      .y(d => y(d.f1))
      .curve(d3.curveMonotoneX);

    const models = Array.from(activeModels);
    const tKey = String(selectedThreshold);

    const paths = g.selectAll<SVGPathElement, string>('.model-line').data(models, d => d);
    paths.enter()
      .append('path').attr('class', 'model-line')
      .attr('fill', 'none').attr('stroke-width', 2.5).attr('opacity', 0)
      .merge(paths)
      .attr('stroke', m => data.meta.modelColors[m])
      .transition().duration(400).ease(d3.easeCubicOut)
      .attr('opacity', 1)
      .attr('d', m => {
        const pts = methods.map(method => ({
          method, f1: data.performance[m]?.[method]?.[tKey]?.F1 ?? 0
        }));
        return line(pts);
      });
    paths.exit().transition().duration(300).attr('opacity', 0).remove();

    const dotData = models.flatMap(m =>
      methods.map(method => ({
        key: `${m}-${method}`,
        model: m, method,
        f1: data.performance[m]?.[method]?.[tKey]?.F1 ?? 0,
      }))
    );
    const dots = g.selectAll<SVGCircleElement, typeof dotData[0]>('.dot').data(dotData, d => d.key);
    dots.enter()
      .append('circle').attr('class', 'dot').attr('r', 4).attr('opacity', 0)
      .on('mouseover', (event, d) => {
        setTip({ x: event.clientX, y: event.clientY, visible: true,
          content: `${d.model}\n${d.method}\nF1: ${d.f1.toFixed(4)}` });
      })
      .on('mouseout', () => setTip(t => ({ ...t, visible: false })))
      .merge(dots)
      .attr('fill', d => data.meta.modelColors[d.model])
      .transition().duration(400).ease(d3.easeCubicOut)
      .attr('cx', d => x(d.method)!)
      .attr('cy', d => y(d.f1))
      .attr('opacity', 1);
    dots.exit().transition().duration(300).attr('opacity', 0).remove();
  });

  return (
    <>
      <ChartContainer
        title="F1 vs Top-K"
        controls={
          <ThresholdSlider
            thresholds={data.meta.thresholds}
            selected={selectedThreshold}
            onChange={onThresholdChange}
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
