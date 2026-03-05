import { useState, useCallback } from 'react';
import { useData } from './hooks/useData';
import { Header } from './components/layout/Header';
import { SummaryStats } from './components/panels/SummaryStats';
import { ModelToggle } from './components/controls/ModelToggle';
import { F1VsThreshold } from './components/charts/F1VsThreshold';
import { F1VsTopK } from './components/charts/F1VsTopK';
import { ScoreDistribution } from './components/charts/ScoreDistribution';
import { RuntimeBreakdown } from './components/charts/RuntimeBreakdown';
import { SaturationDiagnostic } from './components/charts/SaturationDiagnostic';

function App() {
  const { data, loading, error } = useData();
  const [activeModels, setActiveModels] = useState<Set<string>>(new Set());
  const [selectedMethod, setSelectedMethod] = useState('TOP-10');
  const [selectedThreshold, setSelectedThreshold] = useState(0.9);
  const [initialized, setInitialized] = useState(false);

  // Initialize activeModels once data loads
  if (data && !initialized) {
    setActiveModels(new Set(data.meta.models));
    setInitialized(true);
  }

  const toggleModel = useCallback((model: string) => {
    setActiveModels(prev => {
      const next = new Set(prev);
      if (next.has(model)) next.delete(model);
      else next.add(model);
      return next;
    });
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-slate-400 text-lg">Loading dashboard data...</div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-red-400 text-lg">Error: {error ?? 'No data'}</div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <Header />
      <SummaryStats meta={data.meta} />

      <div className="mb-4">
        <ModelToggle
          models={data.meta.models}
          activeModels={activeModels}
          colors={data.meta.modelColors}
          onToggle={toggleModel}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <F1VsThreshold
          data={data}
          activeModels={activeModels}
          selectedMethod={selectedMethod}
          onMethodChange={setSelectedMethod}
        />
        <F1VsTopK
          data={data}
          activeModels={activeModels}
          selectedThreshold={selectedThreshold}
          onThresholdChange={setSelectedThreshold}
        />
        <ScoreDistribution
          data={data}
          activeModels={activeModels}
        />
        <RuntimeBreakdown
          data={data}
          activeModels={activeModels}
        />
        <SaturationDiagnostic
          data={data}
          activeModels={activeModels}
          selectedThreshold={selectedThreshold}
        />
      </div>
    </div>
  );
}

export default App;
