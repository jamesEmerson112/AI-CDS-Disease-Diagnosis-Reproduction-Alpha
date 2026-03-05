export interface MetricRow {
  TP: number;
  FP: number;
  P: number;
  R: number;
  F1: number;
  PR: number;
}

export interface DistributionStats {
  n: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  std: number;
  p5: number;
  p25: number;
  p75: number;
  p95: number;
  thresholdPct: Record<string, number>;
}

export interface TimingData {
  modelLoading: number;
  symptomEmbeddings: number;
  diagnosisEmbeddings: number;
  embeddingsTotal: number;
  foldsProcessing: number;
  totalExecution: number;
}

export interface DashboardMeta {
  patients: number;
  uniqueDiagnoses: number;
  folds: number;
  meanDiagnosesPerPatient: number;
  totalPatientPairs: number;
  models: string[];
  methods: string[];
  thresholds: number[];
  modelColors: Record<string, string>;
}

export interface DashboardData {
  meta: DashboardMeta;
  performance: Record<string, Record<string, Record<string, MetricRow>>>;
  timing: Record<string, TimingData>;
  scoreDistribution: {
    pairwise: Record<string, DistributionStats>;
    perPatientMax: Record<string, DistributionStats>;
  };
  saturation: Record<string, Record<string, number>>;
}
