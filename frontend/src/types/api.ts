export type ApiVersion = "v1";

export interface HealthResponse {
  api_version: ApiVersion;
  status: "ok";
}

export type ReadinessStatus = "ready" | "not_ready";
export type CapacityState = "available" | "full" | "unavailable";

export interface ReadinessResponse {
  api_version: ApiVersion;
  status: ReadinessStatus;
  accepting_jobs: boolean;
  capacity_state: CapacityState;
}

export interface PublicError {
  code: string;
  message: string;
  request_id: string;
}

export interface PublicErrorEnvelope {
  api_version: ApiVersion;
  error: PublicError;
}

export type JobState =
  | "accepted"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "expired";

export interface JobFailure {
  code: string;
  message: string;
  incident_id: string;
}

export interface JobLinks {
  self: string;
  result: string;
}

export interface JobStatus {
  job_id: string;
  state: JobState;
  queue_position: number | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  expires_at: string | null;
  queue_elapsed_ms: number;
  execution_elapsed_ms: number;
  failure: JobFailure | null;
  links: JobLinks;
  poll_after_ms: number | null;
}

export interface JobStatusResponse {
  api_version: ApiVersion;
  job: JobStatus;
}

export interface JobSubmissionResponse {
  api_version: ApiVersion;
  job_id: string;
  state: JobState;
  request_id: string;
}

export type ModelVerdict = "fake" | "real" | "not_run";
export type DisplayVerdict = "Fake" | "Real" | "NEI";
export type EvidenceStatus = "sufficient" | "insufficient";
export type EvidenceSourceType =
  | "text"
  | "transcript"
  | "ocr"
  | "visual_observation";

export interface PublicEvidenceRegion {
  text: string | null;
  bbox: number[];
  confidence: number | null;
}

export type VisualXAIMethod =
  | "qwen_occlusion_logprob_v1"
  | "siglip_semantic_grounding_v1";

export interface PublicVisualXAIMap {
  map_id: string;
  scope: "observation" | "phrase";
  label: string;
  heatmap_image: string | null;
  target_token_count: number;
  baseline_target_log_probability: number;
  raw_importance: number[][];
  normalized_importance: number[][];
}

export interface PublicVisualXAI {
  status: "available" | "unavailable";
  unavailable_reason: string | null;
  method: VisualXAIMethod;
  model_id: string;
  model_revision: string;
  model_fingerprint: string;
  source_frame_sha256: string;
  observation_unit_id: string;
  observation_text_sha256: string;
  raw_generation_sha256: string;
  grid_rows: number;
  grid_columns: number;
  occlusion_baseline: string;
  configuration_version: string;
  phrase_policy: string;
  disclaimer: string;
  scientific_boundary: string;
  attribution_maps: PublicVisualXAIMap[];
}

export interface PublicEvidenceFrame {
  frame_id: string | null;
  frame_index: number | null;
  timestamp: number | null;
  original_image: string | null;
  annotated_image: string | null;
  bbox: number[] | null;
  regions: PublicEvidenceRegion[];
  explanation: string;
  xai?: PublicVisualXAI | null;
}

export interface PublicEvidenceUnit {
  unit_id: string;
  source_type: EvidenceSourceType;
  text: string;
  start_time: number | null;
  end_time: number | null;
  frame_id: string | null;
  bbox: number[] | null;
  confidence: number | null;
  producer: string;
  eligible_for_frozen_g1: boolean;
  selection_score: number | null;
  logits: Record<string, number> | null;
  extraction_method: string;
  source_index: number | null;
  frame_ids: string[];
  evidence_refs: string[];
  source_unit_ids: string[];
  observation_type: string | null;
  evidence_frames?: PublicEvidenceFrame[];
}

export interface PublicVerdict {
  model_verdict: ModelVerdict;
  display_verdict: DisplayVerdict;
  evidence_status: EvidenceStatus;
  sample_logits: Record<string, number>;
  probabilities: Record<string, number>;
  class_winners: Record<string, string>;
  checkpoint_sha256: string | null;
}

export interface PublicSufficiency {
  status: EvidenceStatus;
  reason_code: string;
  model_was_run: boolean;
  g1_exposure_count: number;
  transcript_exposure_count: number;
  ocr_exposure_count: number;
  visual_unit_count: number;
  top_k_count: number;
  supplemental_visual_present: boolean;
}

export interface PublicEvidence {
  g1_exposure_units: PublicEvidenceUnit[];
  g1_top_k_explanation_unit_ids: string[];
  visual_supplemental_units: PublicEvidenceUnit[];
}

export interface ProductionResult {
  schema_version: 1;
  session_id: string;
  claim: string;
  verdict: PublicVerdict;
  sufficiency: PublicSufficiency;
  evidence: PublicEvidence;
  runtime_ms: number;
}

export interface SuccessfulProductionExecutionOutcome {
  schema_version: 1;
  status: "success";
  result: ProductionResult;
  failure: null;
}

export interface JobResultResponse {
  api_version: ApiVersion;
  job_id: string;
  outcome: SuccessfulProductionExecutionOutcome;
}

export interface SubmitJobInput {
  claim: string;
  video: File;
}
