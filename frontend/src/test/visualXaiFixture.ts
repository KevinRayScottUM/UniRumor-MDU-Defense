import type { PublicEvidenceFrame } from "../types";

export const VISUAL_XAI_QA_OBSERVATION =
  "The speaker is positioned centrally on the stage with microphones in front of him.";

export const VISUAL_XAI_QA_ORIGINAL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=";
export const VISUAL_XAI_QA_WHOLE_HEATMAP =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAABSUVORK5CYII=";
export const VISUAL_XAI_QA_PHRASE_HEATMAP =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAABCSUVORK5CYII=";

/** Presentation-only fixture. It never enters request, inference, or API code. */
export const VISUAL_XAI_QA_FRAME: PublicEvidenceFrame = {
  frame_id: "F001",
  frame_index: 25,
  timestamp: 1,
  original_image: VISUAL_XAI_QA_ORIGINAL,
  annotated_image: null,
  bbox: null,
  regions: [],
  explanation: "This is an actual observer source frame.",
  xai: {
    status: "ready",
    unavailable_reason: null,
    method: "qwen_occlusion_logprob_v1",
    model_id: "Qwen/Qwen2.5-VL-7B-Instruct",
    model_revision: "fixture-revision",
    model_fingerprint: "a".repeat(64),
    source_frame_sha256: "b".repeat(64),
    observation_unit_id: "visual-xai-1",
    observation_text_sha256: "c".repeat(64),
    raw_generation_sha256: "d".repeat(64),
    profile: "research",
    grid_rows: 8,
    grid_columns: 8,
    attribution_batch_size: 2,
    occlusion_baseline: "gaussian_blur_region_v1",
    configuration_version: "qwen_occlusion_blur_v2",
    configuration_fingerprint: "e".repeat(64),
    phrase_policy: "deterministic_visible_concept_tokens_v1",
    cache_hit: false,
    queue_wait_ms: 0,
    compute_time_ms: 100,
    source_frame_count: 1,
    heavy_scorer_batches: 33,
    disclaimer:
      "This is a post-hoc perturbation attribution of the Visual Observer. It does not affect the authoritative verification verdict.",
    scientific_boundary:
      "Supplemental visual XAI is explanatory only and does not participate in the Frozen G1 verdict.",
    attribution_maps: [
      {
        map_id: "observation",
        scope: "observation",
        label: "Whole observation",
        heatmap_image: VISUAL_XAI_QA_WHOLE_HEATMAP,
        target_token_count: 14,
        baseline_target_log_probability: -4.2,
        raw_importance: [[1, 0]],
        normalized_importance: [[1, 0]],
      },
      {
        map_id: "phrase_01",
        scope: "phrase",
        label: "Microphones",
        heatmap_image: VISUAL_XAI_QA_PHRASE_HEATMAP,
        target_token_count: 1,
        baseline_target_log_probability: -0.4,
        raw_importance: [[0, 2]],
        normalized_importance: [[0, 1]],
      },
    ],
  },
};
