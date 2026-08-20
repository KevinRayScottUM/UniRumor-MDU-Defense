# Enterprise UI Design System Contract

Status: **Frozen for Task07 implementation**

Product: UniRumor MDU Defense

Primary context: master's thesis defense, academic research showcase, and
controlled public demonstration

This document defines the visual, interaction, responsive, and accessibility
contract for the future React + TypeScript frontend. It contains no CSS, TSX,
React component, or runtime implementation.

## 1. Product and design principles

1. **Evidence before decoration.** The claim, verdict, evidence sufficiency,
   source provenance, and result semantics receive the clearest hierarchy.
2. **Scientific restraint.** The UI presents exact Task06 outcomes without
   invented certainty, thresholds, model stages, or progress percentages.
3. **Auditability through structure.** Full G1 exposure, Top-5 explanation
   selection, supplemental visual observations, and technical details remain
   distinguishable and inspectable.
4. **Calm confidence.** Typography, spacing, motion, and color feel deliberate
   and technically credible rather than theatrical.
5. **Progressive disclosure.** Default views answer “what is the result and
   why?” Advanced logits, class winners, and checkpoint details remain one
   intentional interaction away.
6. **Server truth.** Accepted, queued, running, completed, failed, and expired
   states are rendered only from `/api/v1`; timers and animation never fabricate
   computational progress.
7. **Accessible by default.** Keyboard, screen reader, contrast, reduced motion,
   touch, and responsive behavior are design inputs, not later polish.
8. **One identity in two themes.** Light and dark modes use the same hierarchy,
   semantics, spacing, and motion.

## 2. Visual personality

The target personality is a polished enterprise AI research product:

- credible and evidence-led;
- precise, quiet, and modern;
- thesis-defense ready on a large display;
- dense enough for research inspection but not dashboard-cluttered;
- visually coherent across upload, waiting, result, and failure states; and
- recognizable through disciplined typography, restrained depth, source
  labeling, and an evidence-thread motif rather than a decorative mascot.

Use a mostly neutral canvas with a controlled cool accent. Verdict colors are
semantic signals, not page-wide backgrounds. Depth comes from tonal surfaces,
thin borders, local shadow, and information layering. A subtle evidence-thread
line may connect source summary, explanation units, and detail sections, but it
must never resemble a circuit-board or fake terminal graphic.

## 3. Explicit anti-patterns

The final product must not resemble:

- default Gradio or Streamlit;
- Jupyter/notebook output;
- a generic Bootstrap admin dashboard;
- a university assignment template;
- an unstyled form with one button;
- a card wall with equal emphasis everywhere;
- an obviously AI-generated landing page with oversized gradient text;
- a fake terminal, code rain, cyberpunk HUD, or excessive neon;
- excessive glassmorphism, blur, gradients, or colored glow;
- childish illustrations, confetti, gamified progress, or bouncing status icons;
- a permanent left navigation intended for nonexistent product areas;
- placeholder analytics or fabricated charts; or
- fake ASR/OCR/G1 percentages, completion estimates, or stage indicators.

Do not use color, animation, or a model probability as a substitute for an
explicit label.

## 4. Information architecture

The baseline has two application routes and no account/history dashboard:

```text
/
  Verification workspace
  |-- exact focal claim
  |-- video upload
  |-- method/safety disclosure
  `-- submit

/jobs/:jobId
  Job workspace
  |-- accepted / queued / running state
  |-- completed verdict and sufficiency
  |-- G1 evidence explorer
  |-- Top-5 explanation section
  |-- supplemental visual observations
  `-- technical details
```

The application shell contains:

- compact product mark and “UniRumor MDU Defense” wordmark;
- primary “New verification” action when a job exists;
- restrained method/about disclosure;
- health/connectivity indicator only when actionable;
- theme control; and
- no decorative navigation items or empty enterprise modules.

The job URL is restorable and pollable while the in-memory job exists. The UI
must explain 404-after-restart and expired-job behavior without implying durable
history.

## 5. Primary verification workflow

1. **Land in the verification workspace.** A concise product statement explains
   exact-claim video verification and the distinction between scored text/OCR
   evidence and supplemental visual observation.
2. **Select a video.** Drag/drop and the visible file-picker button are equally
   first-class. Show local filename, size, type, and remove/replace controls.
3. **Enter the exact focal claim.** Explain that the claim is preserved exactly;
   label it explicitly rather than relying on placeholder text.
4. **Validate.** Show field-specific errors before submission without clearing
   valid input.
5. **Submit.** Lock duplicate submission and show real upload/submission state.
   Do not show upload percentage unless the transport measures actual bytes.
6. **Accepted/queued.** Navigate to the job URL. Show server state, actual queue
   position when known, submitted time, and queue elapsed time.
7. **Running.** Replace queue language with a calm execution state and actual
   elapsed time. Do not guess internal stages or completion time.
8. **Reveal completed result.** Present verdict and evidence sufficiency first,
   then source summary and evidence exploration.
9. **Explore evidence.** Allow source filtering, Top-5 explanation review,
   supplemental visual review, and advanced technical disclosure.

The user may start a new verification after a terminal state. A failed or
expired job never silently returns to the upload form; the reason and next
action remain visible.

## 6. Frontend state contract

| Frontend state | Source of truth | Presentation |
| --- | --- | --- |
| `idle` | Browser | Empty claim/file workspace |
| `file_selected` | Browser | Validated local file summary |
| `submitting` | Browser transport | Inputs locked; measured upload or indeterminate submission only |
| `accepted` | Server | Job created; brief acknowledged state |
| `queued` | Server | Queue position if known and real waiting time |
| `running` | Server | Real execution elapsed time, no percentage |
| `completed` | Server | Result workspace |
| `failed` | Server | Operational failure panel, no verdict |
| validation error | Browser/server | Field-associated corrective message |
| network error | Browser transport | Non-destructive connection banner; preserve last server state |
| queue full | Server 429 | Busy state with `Retry-After` behavior; no job link |
| expired | Server 410 | Expired explanation and new-verification action |

Animation never promotes a browser-local assumption to a server state.

## 7. Desktop composition

Desktop (`>= 1024 px`) is the primary thesis-defense experience.

### Verification workspace

Use a centered 12-column composition within the application content width:

- columns 1–5: product explanation, scientific-boundary summary, and compact
  source legend;
- columns 6–12: prominent verification panel containing claim, upload, and
  submit controls; and
- a narrow lower band for privacy, file policy, and processing-time expectation.

The layout must feel like one workspace, not a marketing hero above a form.
Avoid an oversized empty headline. The form panel has the strongest surface,
while the explanatory side remains quiet.

### Queued/running workspace

Use a stable two-region composition:

- a 4-column status rail containing server state, real timers, queue position,
  and request summary; and
- an 8-column calm analysis canvas explaining what will become available after
  completion and the scientific boundaries that remain fixed.

Do not replace the whole page with a centered spinner.

### Result workspace

Use a 12-column analysis layout:

- a sticky 4-column summary rail with verdict hero, sufficiency, model
  probabilities when present, processing time, and source counts; and
- an 8-column evidence workspace with source tabs/filters, Top-5 explanation,
  supplemental visual evidence, and technical disclosure.

The summary rail may stop being sticky near the footer and must never obscure
content at short viewport heights. The evidence region controls the reading
order. Desktop must not merely stack every card vertically.

At very wide widths, preserve readable line length and add outer whitespace;
do not stretch evidence text across the screen.

## 8. Tablet and mobile composition

### Tablet (`768–1023 px`)

- Use an 8-column grid.
- Verification explanation becomes a compact top band; the form uses six to
  eight columns.
- Result summary becomes a full-width horizontal band with verdict, sufficiency,
  timing, and source counts; evidence follows below.
- Source tabs remain visible and horizontally scroll only if labels cannot fit.
- Dialogs may become side drawers when there is enough width.

### Mobile (`< 768 px`)

- Use one deliberate content column, not a desktop grid shrunk until it wraps.
- Keep 16 px minimum page gutters and 44 px minimum touch targets.
- Product header is compact; secondary method content moves into a drawer.
- Verdict, sufficiency, and the primary next action appear before probabilities.
- Evidence filters use a segmented control or accessible horizontal tab list.
- Evidence metadata wraps beneath the source label; no horizontal data tables.
- Technical details use stacked definition lists.
- Dialogs become full-height sheets when required.
- A submit action may be sticky only while the form is valid and must not cover
  errors or the on-screen keyboard.

Mobile remains fully usable, but it does not dictate the richer desktop defense
composition.

## 9. Content width and grid strategy

Semantic layout tokens:

| Token | Role |
| --- | --- |
| `content-reading` | 680 px maximum for prose and evidence text |
| `content-form` | 760 px maximum for single-focus forms |
| `content-application` | 1280 px maximum for normal workspaces |
| `content-defense` | 1440 px maximum for large result presentations |
| `gutter-mobile` | 16 px |
| `gutter-tablet` | 24 px |
| `gutter-desktop` | 32 px |

Desktop uses a 12-column grid with 24 px gutters; tablet uses eight columns with
20 px gutters; mobile uses four conceptual columns with 16 px gutters. Most
mobile compositions span all four columns.

## 10. Typography system

Use a self-hosted variable sans such as **Inter Variable** for product UI and a
self-hosted **IBM Plex Mono** subset only for logits, digests, IDs, and aligned
technical numbers. The fallback stack must remain legible using system sans and
system monospace. External font requests are not required.

| Token | Desktop size / line | Weight | Use |
| --- | --- | --- | --- |
| `type-display` | 48 / 56 px | 600 | One restrained product/result statement |
| `type-title-1` | 36 / 44 px | 600 | Page title or verdict word |
| `type-title-2` | 28 / 36 px | 600 | Major workspace section |
| `type-title-3` | 20 / 28 px | 600 | Card/section title |
| `type-body-lg` | 18 / 28 px | 400 | Lead explanation |
| `type-body` | 16 / 24 px | 400 | Default content and evidence text |
| `type-label` | 14 / 20 px | 550 | Form labels, controls, status labels |
| `type-small` | 14 / 20 px | 400 | Secondary metadata |
| `type-caption` | 12 / 16 px | 500 | Timestamps and compact provenance |
| `type-technical` | 13 / 20 px | 450 mono | IDs, logits, hashes, JSON-like values |

Mobile reduces display to 36/44 and title-1 to 30/38. Evidence text never drops
below 16 px. Use sentence case. Reserve uppercase for short source tags such as
OCR, not headings or paragraphs. Numeric probability and timing values use
tabular numerals. Keep evidence lines near 65–80 characters.

## 11. Spacing system

The base unit is 4 px. Approved spacing tokens are:

```text
space-0  = 0
space-1  = 4
space-2  = 8
space-3  = 12
space-4  = 16
space-5  = 20
space-6  = 24
space-8  = 32
space-10 = 40
space-12 = 48
space-16 = 64
space-20 = 80
```

Use 8–12 px inside dense metadata groups, 16–24 px inside controls/cards, 32–48
px between related sections, and 64–80 px only between major desktop regions.
Do not invent one-off gaps to repair component hierarchy. Mobile major-section
spacing is normally 40–48 px.

## 12. Radius, borders, and dividers

| Token | Value | Use |
| --- | --- | --- |
| `radius-sm` | 6 px | small controls, code values |
| `radius-md` | 10 px | inputs, buttons, evidence rows |
| `radius-lg` | 14 px | major cards and panels |
| `radius-xl` | 18 px | dialogs/drawers only |
| `radius-pill` | full | compact status badges only |

Default borders are 1 px semantic borders. Use a stronger divider for selected
or focus-within surfaces, not a colored glow. Section dividers should align to
content grids and avoid enclosing every text block. Dashed borders are reserved
for the empty upload target. Do not make all containers pill-shaped.

## 13. Surface and elevation system

Surface hierarchy:

| Token | Role |
| --- | --- |
| `surface-canvas` | application background |
| `surface-subtle` | low-emphasis explanatory region |
| `surface-panel` | standard form/evidence container |
| `surface-raised` | active popover, selected evidence, sticky summary |
| `surface-overlay` | dialog/drawer content |
| `surface-scrim` | modal backdrop |

Elevation:

- `elevation-0`: no shadow; use tonal difference/border.
- `elevation-1`: very soft, short card shadow for primary panels.
- `elevation-2`: popover/menu shadow with clear border.
- `elevation-3`: dialog/drawer shadow over a scrim.

Dark mode relies more on surface luminance and borders than shadow. Colored
glows are forbidden. Sticky elements receive a subtle backdrop and divider so
their position remains understandable.

## 14. Semantic color token roles

Implementation must define tokens by semantic role and test their contrast in
both themes. Final values are chosen during visual implementation; components
must never depend on literal color names.

### Foundation

- `color-bg-canvas`
- `color-bg-subtle`
- `color-bg-panel`
- `color-bg-raised`
- `color-bg-overlay`
- `color-text-primary`
- `color-text-secondary`
- `color-text-muted`
- `color-text-inverse`
- `color-border-default`
- `color-border-strong`
- `color-divider`

### Brand and interaction

- `color-accent`
- `color-accent-hover`
- `color-accent-pressed`
- `color-accent-subtle`
- `color-focus-ring`
- `color-selection`
- `color-disabled-bg`
- `color-disabled-text`

### Product semantics

- `color-verdict-fake` and `color-verdict-fake-subtle`
- `color-verdict-real` and `color-verdict-real-subtle`
- `color-verdict-nei` and `color-verdict-nei-subtle`
- `color-status-success` and `color-status-success-subtle`
- `color-status-warning` and `color-status-warning-subtle`
- `color-status-error` and `color-status-error-subtle`
- `color-status-info` and `color-status-info-subtle`
- `color-source-transcript`
- `color-source-ocr`
- `color-source-visual-supplemental`

Fake and Real are classification outcomes, not application error/success. Their
tokens must not reuse operational `status-error` or `status-success` without an
independent mapping review. NEI uses an amber/neutral warning family but is not
styled as a crash. Operational failure uses the error family plus an explicit
failure icon and copy.

No status relies on color alone; pair every semantic color with text, icon, and
where useful a border or pattern.

## 15. Light and dark theme contract

### Light

- Use an off-white/neutral canvas rather than pure white everywhere.
- Panels are slightly brighter than the canvas with visible neutral borders.
- Primary text is near-black, secondary text remains clearly legible.
- Accent and verdict subtle surfaces stay pale enough for dark text.

### Dark

- Use a deep neutral canvas, not pure black.
- Raised surfaces become lighter in ordered steps; borders stay visible.
- Avoid saturated neon accents and bloom effects.
- Verdict surfaces use restrained tint with compliant text and border contrast.

Default follows `prefers-color-scheme`; a user choice overrides it and persists
locally. Theme initialization must avoid a flash of the wrong theme. Theme
switching changes semantic tokens only; layout, hierarchy, icon meanings, and
brand identity remain identical.

## 16. Interaction states

### Focus

- All interactive elements show a 2 px visible semantic focus ring with an
  offset that remains visible against both canvas and panel surfaces.
- `:focus-visible` avoids rings on pointer interaction without suppressing
  keyboard focus.
- Composite widgets follow the correct roving-tabindex or active-descendant
  pattern.

### Hover and pressed

- Hover may shift surface tone, border, or icon by one restrained step.
- Pressed state uses a small tonal change and at most a 1 px visual compression.
- Hover cannot reveal information unavailable to keyboard/touch users.

### Disabled

- Disabled controls remove hover/press motion, use disabled semantic tokens,
  preserve readable labels, and expose the semantic disabled state.
- Opacity alone is insufficient. Explain why submission is unavailable near the
  control when the cause is not obvious.

### Success, warning, and error

- Success confirms accepted upload or completed non-verdict operations; it does
  not label Real as “success.”
- Warning communicates NEI, queue capacity, or recoverable attention without
  implying a crash.
- Error is reserved for validation/transport/operational failure. It never
  contains a Fake/Real/NEI verdict.

## 17. Iconography

Use Lucide icons with consistent 1.75–2 px apparent stroke. Standard sizes are
16 px in dense metadata, 20 px in controls, and 24 px in status headers. Larger
verdict icons may reach 32 px but must not become illustrations.

Icon mapping must remain stable:

- transcript: text/quote icon;
- OCR: scan/text-recognition icon;
- supplemental visual: eye/image icon plus “Supplemental” label;
- queued: clock/list icon;
- running: activity/search icon without progress implication;
- completed: check-in-circle only for job completion, not Real verdict;
- NEI: help/insufficient-evidence icon;
- operational failure: alert triangle;
- technical details: braces/sliders icon.

Icon-only buttons require accessible names and visible tooltips. Never use a
robot/sparkle icon as the sole indicator that content came from a model.

## 18. Motion system

Motion is implemented with Framer Motion-compatible concepts and CSS for simple
states. Use opacity and transforms of 4–8 px; avoid large travel, springy bounce,
3D rotation, parallax, and continuous decorative animation.

### Timing categories

| Category | Duration | Easing | Use |
| --- | --- | --- | --- |
| `motion-micro` | 100–160 ms | standard ease-out | hover, press, focus-adjacent feedback |
| `motion-component` | 180–260 ms | emphasized ease-out | cards, tabs, accordion, upload acceptance |
| `motion-page` | 280–360 ms | decelerating ease-out | page/workspace entry |
| `motion-result` | 360–480 ms | staged decelerating ease-out | terminal result reveal |

### Interaction specifications

- **Page entry:** one short fade + 6 px rise for the main workspace; do not
  animate every text line.
- **Upload drag-over:** border/surface transition within micro timing; no scale
  beyond approximately 1.01.
- **Upload accepted:** file summary fades/slides in and a check icon resolves
  once; no confetti.
- **Button hover/press:** color/border change and at most 1 px press response.
- **Accepted/queued:** one state-label transition; optional low-frequency dot
  pulse indicates waiting but not percentage or stage.
- **Running:** a restrained nonnumeric activity mark may loop slowly; the real
  elapsed timer is the primary feedback.
- **Result reveal:** verdict/sufficiency appears first, summary second, evidence
  navigation third. Total stagger stays under about 240 ms.
- **Evidence cards:** animate only cards entering the viewport/result transition,
  with no repeated scroll spectacle.
- **Tabs/content:** crossfade with 4 px directional shift; preserve container
  stability.
- **Expandable evidence:** animate height/opacity while maintaining readable
  focus and scroll position.
- **Dialog/drawer:** scrim fade plus 8 px/side entrance; focus moves only after
  content is available.
- **Theme:** token colors may transition briefly; typography and layout do not
  animate.

Animation must never suggest that ASR, OCR, visual analysis, or G1 reached a
percentage that the server does not expose.

## 19. Reduced-motion behavior

Honor `prefers-reduced-motion: reduce` on first render and live changes.

- Remove looping pulse, shimmer, parallax, and decorative activity.
- Replace page/result transforms and stagger with an immediate or <= 80 ms
  opacity change.
- Make accordion/dialog state changes immediate while preserving focus.
- Do not smooth-scroll automatically.
- Keep real elapsed time and text status updates; reduction must not hide state.
- User-triggered progress or focus movement remains explicit and predictable.

Reduced motion is an acceptance gate, not an optional variant.

## 20. Upload interaction specification

### Empty state

The upload target contains:

- visible “Choose video” button;
- secondary drag/drop instruction;
- allowed formats (`MP4`, `M4V`, `MOV`, `WebM`);
- configured maximum size as returned/configured for the deployment; and
- concise privacy/temporary-storage copy.

The entire target may open the picker, but it must be a semantic labeled control
and must not trap keyboard events.

### Drag-over and drop

- Drag enter highlights border and surface and announces the drop target.
- Invalid multi-file drop is rejected with a clear error; do not silently take
  the first file.
- Path-looking filenames, unsupported type, empty file, and oversize file show
  specific local errors before submission when detectable.
- Browser validation is assistance; the server remains authoritative.

### Selected state

Show a compact file row with local-only name, human-readable size, media type,
and replace/remove actions. The browser may display the local filename; the API
must not return it. Do not preview or autoplay the video by default.

### Submission

Disable duplicate submit, preserve the exact claim, and show “Uploading and
submitting…” until HTTP 202. A determinate upload bar is allowed only when the
transport owns real bytes-sent/total measurements. Otherwise use an
indeterminate but non-percentage activity label.

On 429 queue full, keep the local claim/file selection, show server-safe busy
copy, honor `Retry-After`, and let the user retry. On network error, preserve
inputs and explain that no job is known unless a 202/job ID was received.

## 21. Queue and running interaction

### Accepted/queued

Required content:

- explicit `Accepted` or `Queued` label from server truth;
- public job ID in a secondary copy control;
- created/submitted timestamp;
- queue elapsed time;
- one-based queue position only when returned by the server; and
- honest copy that inference may take several minutes.

Do not calculate an ETA from queue position. The status rail remains visually
stable as position changes.

### Running

Required content:

- explicit `Running` label from server truth;
- actual execution elapsed time;
- statement that one GPU execution lane is active; and
- explanation that detailed evidence appears after completion.

Do not expose fabricated internal stages, a circular percentage, a nearly full
bar, or rotating claims about which model is currently running. A connection
banner may overlay the last known state, but cannot change it.

## 22. Result reveal and default hierarchy

After `completed`, reveal in this order:

1. verdict hero and structural evidence-sufficiency status;
2. exact focal claim;
3. compact Fake/Real probability presentation when the model ran;
4. processing time and evidence source summary;
5. G1 evidence explorer;
6. Top-5 explanation-only section;
7. supplemental visual observations; and
8. collapsed technical details.

Avoid a full-screen verdict color wash. The verdict hero is a bounded high-
quality surface with text, icon, short explanation, and source context. Result
motion runs once; revisiting a tab does not replay the entire reveal.

## 23. Verdict presentation

### Fake and Real

The hero includes:

- large explicit `Fake` or `Real` label;
- “Frozen G1 binary result” descriptor;
- evidence-sufficiency badge;
- exact focal claim;
- compact two-class probabilities; and
- link/scroll action to explanation evidence.

Probability bars are allowed because they display real returned class
probabilities. They must be labeled “Model class probabilities,” show both Fake
and Real, and never resemble execution progress. Display rounding is permitted
for readability, but exact values remain in technical details and the UI never
recomputes them.

Fake is not styled as an application crash, and Real is not styled as a generic
system success. Their icons, labels, and verdict tokens remain classification-
specific.

## 24. NEI presentation

NEI is a successful completed result and must not look like a crash.

Required default copy:

> Insufficient eligible evidence for a Frozen G1 decision.

The NEI hero uses the NEI semantic family, an insufficient-evidence icon, and
the evidence source summary. It explicitly states:

- the web job completed successfully;
- the model verdict is `not_run`;
- NEI is an engineering abstention display state, not a third learned class;
- no Fake/Real probability or logit is available; and
- supplemental visual observations, if present, cannot independently make
  evidence sufficient.

Do not show empty probability bars, zero-percent probabilities, a retry-as-error
button, or uncertainty-threshold language.

## 25. Operational failure presentation

Operational failure uses a distinct error surface and never renders the verdict
hero. Required content:

- “Verification could not be completed” heading;
- public-safe fixed message and stable failure code;
- public request/correlation ID;
- last known completion time if returned;
- “Start a new verification” action; and
- a retry action only when the client still safely holds the original local file
  or the API later defines an authorized retry contract.

Never show Fake, Real, NEI, probabilities, logits, raw exception messages,
tracebacks, subprocess stderr, internal warnings, or DICC/model/cache/dataset
paths. A failure is not “insufficient evidence.”

## 26. Evidence source summary

The summary appears near the verdict and reports real structural counts:

- G1 exposure total;
- transcript exposure count;
- OCR exposure count;
- Top-5 explanation count (which can be fewer than five); and
- supplemental visual observation count.

Each metric has a text label and source icon. Do not turn counts into
percentages or quality scores. Text units, if present in future results, belong
to G1 exposure but are not mislabeled as transcript/OCR.

## 27. G1 evidence explorer

The explorer defaults to a readable combined chronology and supports transcript
and OCR filters. Each evidence item contains, when present:

- source-type label and icon;
- evidence text;
- start/end time or frame/time reference;
- stable unit ID in secondary details;
- Top-5 explanation marker if its ID is selected; and
- expandable scientific metadata.

OCR bounding-box values and source indices belong in expanded details, not the
primary reading line. Evidence text uses reading-width constraints. Selection
and keyboard focus remain visible when a card expands.

All `g1_exposure_units` remain available; the UI must not display only Top-5 and
call that the prediction evidence.

## 28. Top-5 explanation presentation

The section title is **“Top explanation units”** with a persistent clarification:

> Top-5 is explanation-only. The sample prediction pools class-wise logits over
> every evaluated eligible Frozen G1 unit; selection scores do not define that
> prediction pool.

Render returned IDs in their existing order and resolve each to its G1 evidence
unit. If fewer than five IDs are returned, display the actual count without
empty placeholders. A badge may read `Explanation 1`, `Explanation 2`, and so
on; never `Prediction unit` or `Deciding evidence`.

Selection score may appear only in expanded technical context and must be
labeled “Explanation selection score,” not confidence or probability.

## 29. Supplemental visual evidence presentation

Supplemental visual observations occupy a separate section with a distinct
surface and this persistent contract label:

> Supplemental visual evidence — not scored by Frozen G1.

The persistent user-facing explanation must distinguish retrieval from
observation generation:

> Frames are selected by SigLIP using claim-relevance retrieval. Observation
> text is generated by Qwen claim-blind from the selected frames. Supplemental
> visual evidence is not scored by Frozen G1.

In precise terms, SigLIP frame retrieval is claim-conditioned; only the
selected frames, not the focal claim, are passed to Qwen for claim-blind
observation generation. The resulting visual RuntimeUnits remain supplemental,
`eligible_for_frozen_g1=False`, and carry no Frozen G1 logits, selection score,
or confidence. The UI must never describe the entire visual pipeline as
claim-blind, visually imitate a scored evidence card, add a confidence meter,
or mix visual units into the G1/Top-5 list.

Each observation may show its text, observation type, frame IDs/evidence
references, and time/frame metadata already present in the Task06 public-safe
result. No local frame path is displayed. If no supplemental observation exists,
show a quiet empty state rather than an error.

## 30. Technical details progressive disclosure

Technical details are collapsed by default and open in an accessible accordion
or desktop side drawer. They may expose only existing Task06 public-safe fields:

- schema version;
- exact session/job association;
- model/display verdict and evidence status;
- exact sample logits and probabilities;
- class winners;
- checkpoint SHA-256;
- structural sufficiency reason and counts;
- runtime milliseconds;
- public evidence metadata; and
- exact Top-5 explanation IDs.

Use semantic definition lists or tables with copy controls and monospace only
for values. Raw JSON download/view may be added only from the public-safe
`ProductionExecutionOutcome` response. Never add local paths, internal warnings,
exception details, or reconstructed scientific values.

## 31. Loading, skeleton, and empty states

- Use a short skeleton only while retrieving an existing job's first response.
- Do not run a perpetual shimmer throughout multi-minute inference.
- Running state uses real status/timing content, not placeholder result cards.
- Evidence empty states name the exact absent source, for example “No OCR
  exposure units were returned.”
- A result with no supplemental visuals is complete, not degraded or failed.
- An unknown job uses a neutral not-found page; an expired job uses explicit
  expired copy; neither guesses whether a server restart occurred.
- Preserve layout dimensions while tabs load/transition to avoid large shifts.

## 32. Form validation, transport, and API errors

Errors appear adjacent to their field and in a concise summary only when
multiple fields fail. Required mappings include:

| API condition | UI treatment |
| --- | --- |
| malformed request | Form-level corrective error |
| invalid/blank/long claim | Claim field error |
| empty upload | Upload field error |
| unsupported video | Upload policy error |
| oversized upload | Upload size error with configured limit |
| invalid filename/path syntax | Upload safety error |
| queue full | Busy panel; preserve local input and retry timing |
| service not ready | Availability panel; no job claimed |
| polling network error | Connectivity banner; preserve last server state |
| job expired | Expired page with new-verification action |
| operational failure | Failure page, never NEI |

Use plain public-safe language. Do not print an API response object, stack trace,
or internal error directly into the page.

## 33. Accessibility contract

Target WCAG 2.2 AA for the complete primary workflow.

### Semantics and structure

- One logical `h1`; headings follow a meaningful hierarchy.
- Use `main`, `header`, `nav`, `section`, `form`, `fieldset`, `legend`, `button`,
  and native inputs before ARIA substitutes.
- Evidence lists use list semantics; technical key/value data uses definition
  lists or tables with headers.
- Verdict/status updates use text in the DOM, not canvas-only graphics.

### Keyboard and focus

- Every action is operable with keyboard alone.
- Drag/drop always has a visible keyboard/file-picker path.
- Tab order follows visual/reading order.
- Dialogs/drawers trap focus correctly, close with Escape when safe, and return
  focus to the trigger.
- Route/state changes move focus to the new page heading or a purposeful status
  target without unexpected scrolling.
- Expanded evidence does not remove the triggering control.

### Labels and announcements

- Every form field has a persistent visible label and programmatic description.
- Errors connect through `aria-describedby`; invalid state is programmatic.
- Accepted/queued/running/completed changes use a polite live region.
- Submission failures use an assertive announcement once, without repeated
  polling announcements.
- Icon-only controls have accessible names; decorative icons are hidden from
  assistive technology.

### Visual and motor access

- Normal text contrast is at least 4.5:1; large text and meaningful graphics at
  least 3:1; focus indicators meet WCAG 2.2 contrast/area expectations.
- No information is communicated by color alone.
- Touch/click targets are at least 44 x 44 px where practical, with adequate
  spacing.
- Text supports 200% zoom and browser font scaling without loss of content or
  horizontal page scrolling at 320 CSS px.
- Motion follows the reduced-motion contract.
- Timers do not impose a user response deadline; expired server retention is
  explained but not presented as a countdown requiring fast action.

## 34. Responsive breakpoints and behavior

| Name | Range | Contract |
| --- | --- | --- |
| `compact` | `< 640 px` | single-column mobile; full-width controls/sheets |
| `wide-mobile` | `640–767 px` | single column with richer metadata rows |
| `tablet` | `768–1023 px` | 8-column grid; summary band + evidence |
| `desktop` | `1024–1279 px` | 12-column two-region workspace |
| `defense` | `1280–1439 px` | sticky 4/8 result composition |
| `wide-defense` | `>= 1440 px` | max-width 1440 with intentional outer space |

Breakpoints respond to content pressure, not device names alone. Verify at
390 x 844, 768 x 1024, 1280 x 800, 1440 x 900, and 1920 x 1080. Also verify
short desktop heights, zoom, long claims, long evidence text, and both themes.

## 35. Frontend component taxonomy

Conceptual component groups (names are contracts, not files in Task07A-1):

### Foundations

- `AppShell`, `ContentContainer`, `Grid`, `Stack`, `Cluster`
- `ThemeProvider`, `MotionProvider`
- `Heading`, `Text`, `TechnicalValue`, `Icon`

### Primitives

- `Button`, `IconButton`, `TextArea`, `FilePicker`
- `Badge`, `SourceTag`, `StatusTag`
- `Card`, `Divider`, `Tooltip`, `Popover`
- `Tabs`, `Accordion`, `Dialog`, `Drawer`, `LiveRegion`
- Radix primitives/shadcn-compatible patterns own complex accessibility;
  styling remains project-specific rather than default shadcn appearance.

### Verification feature

- `VerificationWorkspace`
- `ClaimField`
- `VideoDropzone`
- `SelectedFileSummary`
- `SubmissionNotice`

### Job feature

- `JobStatusRail`
- `QueuePosition`
- `ElapsedTime`
- `ConnectivityBanner`
- `ExpiredJobState`

### Result feature

- `VerdictHero`
- `SufficiencyBadge`
- `ClassProbabilityPair`
- `EvidenceSourceSummary`
- `EvidenceExplorer`
- `EvidenceUnitCard`
- `TopExplanationList`
- `SupplementalVisualSection`
- `TechnicalDetails`
- `OperationalFailureState`

Components consume typed `/api/v1` data and semantic tokens. No component may
call a model, recalculate a verdict, or infer a backend state from time.

## 36. Enterprise visual quality acceptance criteria

The frontend passes this contract only when all criteria below are verified:

### Identity and composition

- The result desktop uses the deliberate summary/evidence split, not a vertical
  pile of equal cards.
- The upload page reads as an evidence product workspace, not a plain form or
  generic landing template.
- Typography, spacing, radius, border, shadow, icon, and color tokens are
  consistently applied.
- Light and dark modes clearly belong to the same product.
- No default Gradio, Streamlit, Bootstrap, browser, Radix, or shadcn styling is
  visible as the product identity.

### Scientific honesty

- Fake, Real, NEI, and operational failure are four unmistakably different
  presentations.
- Successful NEI says “Insufficient eligible evidence for a Frozen G1
  decision” and never appears as failure.
- Top-5 is labeled explanation-only and the UI states that prediction pools
  every evaluated eligible unit.
- Supplemental visual evidence is separate and explicitly not scored by Frozen
  G1.
- Visual copy states that SigLIP retrieval is claim-conditioned/
  claim-relevance-based while Qwen observation generation is claim-blind; it
  never labels the entire visual pipeline claim-blind.
- No fake model-stage percentage, confidence, ETA, or visual score appears.

### Interaction quality

- Upload, accepted, queued, running, completed, failed, queue-full, expired, and
  network-error flows are intentionally designed.
- Motion follows the timing system and never fabricates progress.
- Reduced-motion mode removes nonessential motion without losing information.
- Focus, hover, pressed, disabled, success, warning, and error states are
  complete in both themes.

### Responsive and accessible quality

- Large desktop, laptop, tablet, and mobile layouts are composed intentionally.
- Full workflow is keyboard operable with visible focus and correct focus
  movement.
- Labels, live regions, error association, contrast, touch targets, zoom, and
  screen-reader names meet the accessibility contract.
- Long claims/evidence, no OCR, no visuals, fewer than five Top explanations,
  NEI, and operational failure do not break layout or meaning.

### Technical credibility

- The UI exposes exact public-safe technical values through progressive
  disclosure without overwhelming the default result.
- Local paths, raw logs, exception details, and internal warnings never appear.
- The frontend remains a static React + TypeScript deployment and uses the
  configured FastAPI `/api/v1` origin.
- The finished interface is credible on a master's thesis-defense display and
  strong enough for an academic portfolio screenshot without hiding its
  scientific limitations.
