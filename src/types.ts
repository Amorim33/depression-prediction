export type BinaryLabel = "diagnosed" | "control";
export type SplitName = "train" | "test";

export interface DbTables {
  trainUserEmb: string;
  testUserEmb: string;
  trainUserEmbRel3: string;
  testUserEmbRel3: string;
  trainSubFeatures: string;
  testSubFeatures: string;
  trainEmbeddings: string;
  testEmbeddings: string;
}

export interface ProjectConfig {
  dataset: "setembrobr";
  seed: number;
  foldCount: number;
  outputDir: string;
  database: {
    tables: DbTables;
    embeddingDimension: number;
  };
  models: string[];
  ensemble: {
    weightStep: number;
    primaryMetric: "macroF1";
    tieBreakers: Array<"diagnosedF1" | "precision">;
  };
}

export interface ManifestRow {
  dataset: "setembrobr";
  split: SplitName;
  label: BinaryLabel;
  userId: string;
  rowHash: string;
  fold: number | null;
}

export interface ScoreRow {
  userId: string;
  label?: BinaryLabel;
  fold?: number;
  score: number;
  modelId: string;
}

export interface Metrics {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  precision: number;
  recall: number;
  diagnosedF1: number;
  controlF1: number;
  macroF1: number;
  accuracy: number;
}

export interface EnsembleLock {
  dataset: "setembrobr";
  seed: number;
  manifestHash: string;
  modelIds: string[];
  weights: Record<string, number>;
  threshold: number;
  oofMetrics: Metrics;
  sourceHashes: Record<string, string>;
  createdAt: string;
  command: string;
}

