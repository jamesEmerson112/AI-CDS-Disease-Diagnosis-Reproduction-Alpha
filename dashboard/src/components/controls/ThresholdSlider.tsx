interface Props {
  thresholds: number[];
  selected: number;
  onChange: (threshold: number) => void;
}

export function ThresholdSlider({ thresholds, selected, onChange }: Props) {
  const idx = thresholds.indexOf(selected);

  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-400">Threshold:</span>
      <input
        type="range"
        min={0}
        max={thresholds.length - 1}
        step={1}
        value={idx}
        onChange={(e) => onChange(thresholds[Number(e.target.value)])}
        className="w-32 accent-sky-400"
      />
      <span className="text-sm font-mono text-slate-200 w-8">{selected}</span>
    </div>
  );
}
