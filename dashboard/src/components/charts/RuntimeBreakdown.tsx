import { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import type { DashboardData } from '../../types';
import { ChartContainer } from '../shared/ChartContainer';
import { RUNTIME_COLORS } from '../../utils/colors';
import { Tooltip } from '../shared/Tooltip';

interface Props {
  data: DashboardData;
  activeModels: Set<string>;
}

export function RuntimeBreakdown({ data, activeModels }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tip, setTip] = useState({ x: 0, y: 0, visible: false, content: '' });

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth || 500;
    const height = svgRef.current.clientHeight || 320;
    const margin = { top: 20, right: 20, bottom: 35, left: 55 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const models = Array.from(activeModels).filter(m => data.timing[m]);

    const segments = models.map(m => {
      const t = data.timing[m];
      return {
        model: m,
        loading: t.modelLoading / 60,
        embeddings: t.embeddingsTotal / 60,
        folds: t.foldsProcessing / 60,
        total: t.totalExecution / 60,
      };
    });

    const x = d3.scaleBand<string>().domain(models).range([0, w]).padding(0.35);
    const yMax = d3.max(segments, d => d.total) ?? 1;
    const y = d3.scaleLinear().domain([0, yMax * 1.15]).range([h, 0]);

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
      .call(d3.axisLeft(y).ticks(5).tickFormat(d => `${d}m`))
      .selectAll('text,line,path').attr('stroke', '#64748b').attr('fill', '#94a3b8');

    // Stacked bars
    type SegmentType = { key: string; model: string; y0: number; y1: number; color: string; label: string; value: number };
    const barData: SegmentType[] = segments.flatMap(d => {
      let cumY = 0;
      return [
        { key: `${d.model}-load`, model: d.model, y0: cumY, y1: (cumY += d.loading), color: RUNTIME_COLORS.modelLoading, label: 'Model Loading', value: d.loading },
        { key: `${d.model}-emb`, model: d.model, y0: cumY, y1: (cumY += d.embeddings), color: RUNTIME_COLORS.embeddings, label: 'Embeddings', value: d.embeddings },
        { key: `${d.model}-fold`, model: d.model, y0: cumY, y1: (cumY += d.folds), color: RUNTIME_COLORS.foldsProcessing, label: '10-Fold Eval', value: d.folds },
      ];
    });

    const bars = g.selectAll<SVGRectElement, SegmentType>('.bar-seg').data(barData, d => d.key);
    bars.enter()
      .append('rect').attr('class', 'bar-seg')
      .attr('x', d => x(d.model)!)
      .attr('width', x.bandwidth())
      .attr('y', h).attr('height', 0)
      .attr('rx', 2)
      .on('mouseover', (event, d) => {
        setTip({ x: event.clientX, y: event.clientY, visible: true,
          content: `${d.model}\n${d.label}: ${d.value.toFixed(1)} min` });
      })
      .on('mouseout', () => setTip(t => ({ ...t, visible: false })))
      .merge(bars)
      .attr('fill', d => d.color)
      .transition().duration(600).ease(d3.easeCubicOut)
      .attr('x', d => x(d.model)!)
      .attr('width', x.bandwidth())
      .attr('y', d => y(d.y1))
      .attr('height', d => Math.max(0, y(d.y0) - y(d.y1)));
    bars.exit().transition().duration(300).attr('height', 0).attr('y', h).remove();

    // Total labels
    const totalLabels = g.selectAll<SVGTextElement, typeof segments[0]>('.total-label')
      .data(segments, d => d.model);
    totalLabels.enter().append('text').attr('class', 'total-label')
      .merge(totalLabels)
      .transition().duration(600)
      .attr('x', d => x(d.model)! + x.bandwidth() / 2)
      .attr('y', d => y(d.total) - 6)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8').attr('font-size', '11px')
      .text(d => `${d.total.toFixed(1)}m`);
    totalLabels.exit().remove();

    // Legend
    g.selectAll('.legend-item').remove();
    const legendItems = [
      { label: 'Model Loading', color: RUNTIME_COLORS.modelLoading },
      { label: 'Embeddings', color: RUNTIME_COLORS.embeddings },
      { label: '10-Fold Eval', color: RUNTIME_COLORS.foldsProcessing },
    ];
    legendItems.forEach((item, i) => {
      const ly = i * 18;
      g.append('rect').attr('class', 'legend-item')
        .attr('x', w - 110).attr('y', ly).attr('width', 12).attr('height', 12)
        .attr('fill', item.color).attr('rx', 2);
      g.append('text').attr('class', 'legend-item')
        .attr('x', w - 93).attr('y', ly + 10)
        .attr('fill', '#94a3b8').attr('font-size', '10px')
        .text(item.label);
    });
  });

  return (
    <>
      <ChartContainer title="Runtime Breakdown">
        {({ width, height }) => (
          <svg ref={svgRef} width={width} height={height} />
        )}
      </ChartContainer>
      <Tooltip {...tip} />
    </>
  );
}
