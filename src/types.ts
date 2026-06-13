export type BinaryLabel = "diagnosed" | "control";
export type TernaryLabel = "diagnosed" | "control" | "no-evidence";
export type SplitName = "train" | "test";

export interface DbTables {
  trainUserEmb: string;
  testUserEmb: string;
  trainUserEmbRel3: string;
  testUserEmbRel3: string;
  trainUserEmbRel6?: string;
  testUserEmbRel6?: string;
  trainUserEmbRel7?: string;
  testUserEmbRel7?: string;
  trainSubFeatures: string;
  testSubFeatures: string;
  trainV2SubFeatures?: string;
  testV2SubFeatures?: string;
  trainRel5CombinedFeatures?: string;
  testRel5CombinedFeatures?: string;
  trainEmbeddings: string;
  testEmbeddings: string;
}

export type TabularCandidateFamily = "logreg" | "focal_logreg" | "mlp" | "extra_trees";
export type SequenceCandidateFamily = "cnn" | "cnn_wide" | "bilstm" | "tiny_transformer";

export interface TabularCandidateModel {
  modelId: string;
  family: TabularCandidateFamily;
  seed: number;
  featureBlocks: string[];
  pcaComponents?: number;
  gamma?: number;
  hiddenSize?: number;
  alpha?: number;
  nEstimators?: number;
  maxDepth?: number;
  minSamplesLeaf?: number;
}

export interface SequenceCandidateModel {
  modelId: string;
  family: SequenceCandidateFamily;
  seed: number;
  topN: number;
  epochs?: number;
  batchSize?: number;
  numFilters?: number;
  hiddenSize?: number;
  dropout?: number;
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
  candidateModels?: {
    tabular: TabularCandidateModel[];
    sequence: SequenceCandidateModel[];
  };
  ensemble: {
    weightStep: number;
    primaryMetric: "macroF1";
    tieBreakers: Array<"diagnosedF1" | "precision">;
    exhaustiveModelLimit?: number;
    candidatePruneTo?: number;
    maxModels?: number;
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
  selectionStrategy?: string;
}

export interface EvidenceMarker {
  userId: string;
  totalTweets: number;
  maxRelevance: number;
  rel3Count: number;
  rel5Count: number;
  rel6Count: number;
  rel7Count: number;
  rel3Ratio: number;
  rel5Ratio: number;
  rel6Ratio: number;
  rel7Ratio: number;
  top10AvgRelevance: number;
  evidenceScore: number;
}

export interface TernaryManifestRow {
  dataset: "setembrobr";
  split: "train";
  label: TernaryLabel;
  binaryLabel: BinaryLabel;
  userId: string;
  rowHash: string;
  fold: number;
  labelPolicyId: string;
}

export interface TernaryProbabilityRow {
  userId: string;
  label?: TernaryLabel;
  fold?: number;
  probDiagnosed: number;
  probControl: number;
  probNoEvidence: number;
  modelId: string;
  labelPolicyId: string;
}

export interface TernaryClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface TernaryMetrics {
  macroF1: number;
  accuracy: number;
  diagnosedF1: number;
  diagnosedPrecision: number;
  diagnosedRecall: number;
  perClass: Record<TernaryLabel, TernaryClassMetrics>;
  confusion: Record<TernaryLabel, Record<TernaryLabel, number>>;
}

export interface TernaryDecisionRule {
  ruleId: string;
  kind: "argmax" | "diagnosed_margin" | "no_evidence_gate";
  diagnosedMargin?: number;
  noEvidenceMin?: number;
}

export interface TernaryLabelPolicyConfig {
  policyId: string;
  kind: "rel_count_zero" | "low_density" | "top10_avg_lt" | "evidence_quantile";
  relevanceThreshold?: 3 | 5 | 6 | 7;
  densityThreshold?: number;
  top10AvgThreshold?: number;
  quantile?: number;
  description: string;
}

export interface TernaryLabelPolicyLock extends TernaryLabelPolicyConfig {
  dataset: "setembrobr";
  seed: number;
  originalManifestHash: string;
  evidenceFormulaVersion: "v1";
  cutoff?: number;
  policyHash: string;
  createdAt: string;
}

export type TernaryTabularCandidateFamily =
  | "multinomial_logreg"
  | "mlp"
  | "extra_trees"
  | "xgboost"
  | "hist_gradient_boosting"
  | "focal_linear"
  | "hierarchical_logreg"
  | "relevance_baseline";

export interface TernaryTabularCandidateModel {
  modelId: string;
  family: TernaryTabularCandidateFamily;
  seed: number;
  featureBlocks: string[];
  pcaComponents?: number;
  gamma?: number;
  hiddenSize?: number;
  alpha?: number;
  nEstimators?: number;
  maxDepth?: number;
  minSamplesLeaf?: number;
  maxIter?: number;
  maxLeafNodes?: number;
  l2Regularization?: number;
  learningRate?: number;
  subsample?: number;
  colsampleBytree?: number;
  regLambda?: number;
  minChildWeight?: number;
}

export interface TernarySequenceCandidateModel {
  modelId: string;
  family: SequenceCandidateFamily;
  seed: number;
  topN: number;
  useRelevanceChannel?: boolean;
  epochs?: number;
  batchSize?: number;
  numFilters?: number;
  hiddenSize?: number;
  dropout?: number;
}

export interface TernaryModelSelectionGroup {
  groupId: string;
  description: string;
  modelIds: string[];
}

export interface TernaryStackingCandidateModel {
  modelId: string;
  family: "stacking_logreg";
  seed: number;
  baseModelIds: string[];
  c?: number;
  maxIter?: number;
}

export interface TernaryProjectConfig {
  dataset: "setembrobr";
  seed: number;
  foldCount: number;
  sourceOutputDir: string;
  outputDir: string;
  database: {
    tables: DbTables;
    embeddingDimension: number;
  };
  labelPolicies: TernaryLabelPolicyConfig[];
  models: string[];
  candidateModels: {
    tabular: TernaryTabularCandidateModel[];
    sequence: TernarySequenceCandidateModel[];
    stacking?: TernaryStackingCandidateModel[];
  };
  ensemble: {
    weightStep: number;
    primaryMetric: "macroF1";
    tieBreakers: Array<"diagnosedF1" | "diagnosedPrecision">;
    exhaustiveModelLimit?: number;
    candidatePruneTo?: number;
    maxModels?: number;
    decisionRules: TernaryDecisionRule[];
    selectionGroups?: TernaryModelSelectionGroup[];
  };
  documentation?: {
    protocolPath?: string;
    resultsPath?: string;
  };
}

export interface TernaryEnsembleLock {
  dataset: "setembrobr";
  seed: number;
  originalManifestHash: string;
  labelPolicyId: string;
  labelPolicyHash: string;
  modelIds: string[];
  weights: Record<string, number>;
  decisionRule: TernaryDecisionRule;
  oofMetrics: TernaryMetrics;
  sourceHashes: Record<string, string>;
  createdAt: string;
  command: string;
  selectionStrategy?: string;
  selectionGroupId?: string;
  selectionGroupDescription?: string;
  candidateModelIds?: string[];
}
