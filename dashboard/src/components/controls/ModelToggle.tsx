interface Props {
  models: string[];
  activeModels: Set<string>;
  colors: Record<string, string>;
  onToggle: (model: string) => void;
}

export function ModelToggle({ models, activeModels, colors, onToggle }: Props) {
  return (
    <div className="flex flex-wrap gap-3">
      {models.map((model) => (
        <label
          key={model}
          className="flex items-center gap-2 cursor-pointer select-none text-sm"
        >
          <span
            className="w-3 h-3 rounded-full inline-block shrink-0"
            style={{
              backgroundColor: activeModels.has(model) ? colors[model] : '#475569',
              opacity: activeModels.has(model) ? 1 : 0.4,
            }}
          />
          <input
            type="checkbox"
            className="sr-only"
            checked={activeModels.has(model)}
            onChange={() => onToggle(model)}
          />
          <span className={activeModels.has(model) ? 'text-slate-200' : 'text-slate-500'}>
            {model}
          </span>
        </label>
      ))}
    </div>
  );
}
