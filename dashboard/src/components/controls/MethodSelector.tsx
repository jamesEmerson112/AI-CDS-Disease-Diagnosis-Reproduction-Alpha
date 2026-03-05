interface Props {
  methods: string[];
  selected: string;
  onChange: (method: string) => void;
}

export function MethodSelector({ methods, selected, onChange }: Props) {
  return (
    <select
      value={selected}
      onChange={(e) => onChange(e.target.value)}
      className="bg-slate-700 text-slate-200 text-sm rounded px-2 py-1 border border-slate-600 focus:outline-none focus:border-sky-400"
    >
      {methods.map((m) => (
        <option key={m} value={m}>{m}</option>
      ))}
    </select>
  );
}
