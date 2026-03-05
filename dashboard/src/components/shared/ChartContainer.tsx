import { useRef, useEffect, useState, type ReactNode } from 'react';

interface Props {
  title: string;
  controls?: ReactNode;
  children: (dims: { width: number; height: number }) => ReactNode;
}

export function ChartContainer({ title, controls, children }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ width: 500, height: 320 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const { width } = entries[0].contentRect;
      if (width > 0) setDims({ width, height: 320 });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
        {controls && <div className="flex items-center gap-3">{controls}</div>}
      </div>
      <div ref={containerRef} className="w-full">
        {children(dims)}
      </div>
    </div>
  );
}
