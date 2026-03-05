import { useEffect, useState } from 'react';
import type { DashboardMeta } from '../../types';

interface Props {
  meta: DashboardMeta;
}

function AnimatedNumber({ target, suffix = '' }: { target: number; suffix?: string }) {
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    const duration = 800;
    const start = performance.now();
    const step = (now: number) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      setCurrent(Math.round(target * eased * 100) / 100);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [target]);

  const display = Number.isInteger(target) ? Math.round(current) : current.toFixed(2);
  return <>{display}{suffix}</>;
}

export function SummaryStats({ meta }: Props) {
  const stats = [
    { label: 'Patients', value: meta.patients },
    { label: 'Unique Diagnoses', value: meta.uniqueDiagnoses },
    { label: 'Avg Dx/Patient', value: meta.meanDiagnosesPerPatient },
    { label: 'CV Folds', value: meta.folds },
    { label: 'BERT Models', value: meta.models.length },
    { label: 'Patient Pairs', value: meta.totalPatientPairs },
  ];

  return (
    <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-6">
      {stats.map(({ label, value }) => (
        <div
          key={label}
          className="bg-slate-800 rounded-lg p-3 border border-slate-700 text-center"
        >
          <div className="text-xl font-bold text-sky-400">
            <AnimatedNumber target={value} />
          </div>
          <div className="text-xs text-slate-400 mt-1">{label}</div>
        </div>
      ))}
    </div>
  );
}
