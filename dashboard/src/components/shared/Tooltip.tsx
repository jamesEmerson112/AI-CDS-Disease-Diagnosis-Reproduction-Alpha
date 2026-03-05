interface Props {
  x: number;
  y: number;
  visible: boolean;
  content: string;
}

export function Tooltip({ x, y, visible, content }: Props) {
  if (!visible) return null;
  return (
    <div
      className="fixed pointer-events-none z-50 bg-slate-900 text-slate-200 text-xs px-2.5 py-1.5 rounded shadow-lg border border-slate-600 whitespace-pre-line"
      style={{ left: x + 12, top: y - 10 }}
    >
      {content}
    </div>
  );
}
