# Reusable Internal Computation in Language Models: An ABC Test with Attribution Graphs and Causal Interventions

**Working Article Draft**  
**Status:** research manuscript under revision; not yet a submission claim  
**Target format:** Nature Machine Intelligence Article

## Abstract

Large language models solve related problems in different forms, but it is unclear whether this reflects reusable computation or generic answer machinery. We introduce an ABC benchmark expressing one computation explicitly, as an applied word problem, or implicitly in a structured record, with controls that preserve wording or answer identity while changing the operation. We combine behavioral transfer, representation analysis, attribution graphs, sparse interventions, multilayer paths, attention-source tests and component ablations. Gemma 2 2B graphs recur strongly within matched prompt forms, and graph-selected paths and source positions influence answer margins. However, discovery-selected recurrent features do not show held-out necessity relative to active same-position controls, and activation transfer does not produce operation-specific sufficiency. The results support recurring answer-directed computation, but not a discrete reusable reasoning module. They provide a reproducible benchmark and falsification-oriented pipeline for testing stronger modularity claims.

## Introduction

Understanding how language models compute is important for both science and reliability. A model may answer two questions correctly for superficially different reasons: it may retrieve memorized text, use a generic numerical answer pathway, or reuse an internal computation that generalizes across contexts. Distinguishing these possibilities requires more than comparing outputs. It requires matched behavioral controls, internal measurements and interventions on the mechanisms that connect inputs to answer logits.

Recent interpretability work has developed sparse features and attribution graphs as tools for this purpose. Dictionary-learning methods represent dense neural activity with recurring sparse features, while attribution graphs connect feature nodes across layers to output logits. In a mental-math case study, Anthropic reported parallel computational paths for approximate magnitude and precise last-digit information, together with causal interventions that changed the model's answer behavior. These results motivate a smaller, testable question: when a model encounters the same abstract computation in different surface forms, does it recruit a recurring causal pathway?

We study this question with an ABC benchmark. A prompt states a computation explicitly, B expresses it as a short applied problem, and C requires the operation from an implicit structured record. The benchmark is paired with same-surface/different-computation, same-answer/different-computation, near-miss and unrelated controls. The design separates four evidence levels. Behavior asks whether a worked example transfers. Representation asks whether internal states or sparse features recur. Intervention asks whether changing a candidate changes an answer objective. Circuit analysis asks whether a multilayer, component-level path can be traced and transferred.

Our central result is deliberately narrower than a module claim. Gemma 2 2B produced recurring answer-directed graph structure and positive in-sample path and attention-source effects. Yet the stronger held-out test did not show that discovery-selected recurrent sparse features were necessary for addition, and source activation transfer did not show operation-specific sufficiency. Larger Gemma 2 9B and Qwen3 8B models showed positive same-computation residual transfer for addition, but those tests used whole residual states and did not provide sparse circuit localization. Thus, the project establishes a functioning cross-level test of reusable computation while showing why graph recurrence alone is insufficient.

## Results

### The ABC benchmark separates computation from wording

Each latent arithmetic problem was rendered in three forms. For addition, an explicit prompt was `calc: 1+2=`, an applied prompt was `A shelf has 1 books and receives 2 more. How many books now?`, and an implicit prompt was `Record: start=1; added=2; total=?`. The same structure was used for subtraction, multiplication and other simple operations. Controls used similar wording with a different operation, the same answer produced by a different operation, a one-number near miss, or an unrelated prompt.

The early multi-model program found positive aggregate behavioral transfer (+0.0926, 95% CI [0.0309, 0.1667]) and positive late-layer alignment (+0.0509, 95% CI [0.0390, 0.0632]) in a broad benchmark. Sparse-feature reuse was also positive in that exploratory package (+0.2297, 95% CI [0.1737, 0.2879]). These results motivated the stricter circuit audit, but they were not treated as proof of a mechanism because residual patching and held-out component tests were inconclusive or null.

### Attribution graphs recur within prompt forms

We generated 28 Gemma 2 2B attribution graphs with the local Gemma Scope replacement-model pipeline: 24 addition graphs covering eight examples in each ABC form, plus four subtraction and multiplication controls. Every graph completed. For each answer target, we retained the top 80 sparse-feature parents, producing 2,240 parent rows.

The top-parent sets had mean Jaccard overlap of 0.6104 across 84 same-form addition pairs. The mean overlap across 192 different-form addition pairs was 0.1067. The same-form/different-form difference is large, but its interpretation is limited. The graph procedure uses a bounded top-parent set, and same-form prompts share token layout and surface structure. The result shows structured recurrence in the graph representation, not yet operation-specific modularity.

The recurrence table contained 449 layer-feature pairs. Several late-layer features appeared in nearly every addition graph, including `L25_F4717`, `L24_F13541` and `L25_F13822`. However, many highly recurrent features also appeared in subtraction or multiplication graphs. This is a warning against equating frequency with mechanism.

### Multilayer paths and attention source positions influence answer margins

To test whether graph structure had causal relevance, we selected the eight highest-scoring paths in each of eight explicit addition graphs. A path is a sequence of sparse feature nodes at recorded layers and token positions leading toward the answer logit. Zeroing the feature nodes of a path reduced the gold-minus-foil answer margin by 0.6603 on average across 64 successful interventions (95% CI [0.3254, 0.9951]). Boosting the same nodes produced a mean gain of 0.3484 (95% CI [0.0857, 0.6111]).

We separately decomposed four individual attention heads by query-key score, attention weight, output-value projection and source position. Removing one selected source position from a head's attention pattern at the final query position reduced the gold-minus-foil margin by 0.2065 on average across 128 successful tests (95% CI [0.1593, 0.2537]). Attention weight correlated modestly with source-position damage (r = 0.3000). These results establish that the implementation reaches individual heads, source positions and multilayer feature paths. They do not establish that the tested paths are addition-specific, because this proof-of-concept was in-sample and selected from the same prompt panel.

### Held-out recurrent features do not pass the necessity test

The decisive test froze sparse candidates using only the first four examples in each ABC form. Candidates had to recur in at least three discovery graphs and at least two forms. The top 12 candidates were then tested on the final four examples in each form at exact graph-reported positions. Each recurrent feature was compared with a random feature that was active at the same layer and position. This active-control correction matters: arbitrary feature IDs are often already zero and are not valid intervention controls.

All 564 interventions succeeded. Zeroing recurrent features produced mean damage of -0.0122 (95% CI [-0.0266, 0.0021]), whereas zeroing active controls produced +0.0092 (95% CI [-0.0117, 0.0301]). The paired recurrent-minus-control difference was -0.0214 (95% CI [-0.0485, 0.0056]). Negative damage means that zeroing often improved the answer margin. Negative-original replacement gave the same qualitative result. Consequently, discovery recurrence did not predict held-out necessity, and recurrent features did not outperform active controls.

### Activation transfer does not establish sufficiency

We tested whether recurrent feature activations could be transplanted from one addition prompt to another with the same answer, and then from explicit addition into applied, implicit, subtraction and multiplication targets. In same-answer commutative pairs, six-feature chains had mean zeroing damage of -0.5410 and mean transfer gain relative to zeroing of -0.5352. The apparent near-complete rescue fraction is not evidence of sufficiency because zeroing generally improved the clean target margin; transfer mostly returned the target toward its original state.

In the cross-form panel, source-transfer damage was -0.0076 for applied addition, -0.0053 for implicit addition, +0.0026 for subtraction and -0.0029 for multiplication. The transfer-minus-zero differences were not selectively larger for addition. The current sparse candidates are therefore causally manipulable but do not yet behave like a portable addition computation.

### Larger models show residual compatibility, not circuit localization

We ran the corrected residual-patching battery on Qwen3 8B and Gemma 2 9B. In Gemma 2 9B, replacing a target's selected residual state with a same-computation addition source increased the gold-minus-foil score by +1.991 (95% CI [+1.520, +2.463]). The surface/wrong-computation control was -0.767 (95% CI [-1.628, +0.094]) and the unrelated control was -0.176 (95% CI [-1.012, +0.660]). Qwen3 8B showed the same direction for addition (+1.528, 95% CI [+0.952, +2.104]), with a near-zero surface control and a negative unrelated control.

These larger-model results are useful evidence that related arithmetic states can be compatible at selected residual locations. They are not attribution-graph results: the representation is a whole residual vector, and the tested Gemma 9B/Qwen3 models did not have a validated sparse-transcoder graph bundle in this project. The component screen found strong late-MLP ablation effects in Gemma 9B but near-zero selected-head effects. This points to distributed late computation rather than an identified individual-head circuit.

## Discussion

The experiments produce a coherent but bounded picture. Answer-directed sparse paths and attention source routes exist and can influence arithmetic answer margins. Same-form graph structure is highly recurrent. Larger models show positive same-computation residual compatibility, with the cleanest addition contrast in Gemma 2 9B. These findings justify studying reusable internal computation as a serious question.

The stricter causal results prevent the stronger conclusion that the model contains a reusable addition module. Graph recurrence may reflect generic answer formatting, number processing, prompt-template structure, compensatory downstream activity or selection artifacts. A feature can be enriched without being necessary, and a path can be causally influential without being the algorithm. The held-out active-control result is therefore central, not a disappointing footnote.

The study also clarifies what a successful future demonstration would require. Candidates must be selected on behaviorally mastered discovery prompts, validated on held-out prompts with active matched controls, and tested for both necessity and sufficiency. A full circuit claim should additionally map the sparse feature path to attention heads and MLP contributions, test source-token routing and edge-level interventions, and show operation-specific transfer into new surface forms and unrelated controls. Replication across model families and scales is needed before calling the mechanism reusable.

The present benchmark and pipeline make those requirements explicit. The result is best viewed as a falsification-oriented foundation: it detects positive associative and path-level signals, then tests whether they survive the stronger causal controls. At present they do not. That is scientifically useful because it narrows the hypothesis from "LLMs contain reusable reasoning modules" to the more defensible possibility that they reuse distributed late-stage computation whose algorithmic specificity remains unresolved.

## Methods

### Models and computation

The primary graph and component analysis used `google/gemma-2-2b-it` in FP16 on an NVIDIA L4. The model was loaded from the local cache and paired with Gemma Scope replacement-model transcoders at 26 layers. Larger residual screens used Qwen3 8B and Gemma 2 9B in FP16 on the same cloud GPU. No quantization was used in the reported larger-model screens.

### Objectives and tokenization

The main causal objective was the logit margin between the exact gold answer token and a foil token. Gold and foil IDs were derived from the actual tokenization of each prompt and answer. Prompts whose gold answers were not single tokens were excluded from the small causal panels or redesigned before rerunning. All intervention effects are reported as baseline margin minus changed margin; positive values mean that the intervention reduced preference for the gold answer.

### Attribution graphs

For each prompt, the attribution tool targeted the answer logit and traced contributions through the replacement-model feature graph. Graphs used a bounded node budget and retained the strongest answer-target parents. A graph parent is an attribution edge, not a causal effect. Pairwise graph overlap used Jaccard similarity of the retained parent feature sets.

### Path and component interventions

Path tests zeroed or boosted the sparse feature nodes along a graph-selected multilayer path at their recorded layer and position. Attention tests captured individual head patterns and removed one source position from the selected head's attention pattern at the final query position. MLP and attention-head ablations modified the corresponding component output at the final answer position. These operations were performed one prompt at a time with memory checks and saved raw rows for every successful intervention.

### Discovery and confirmation

The held-out sparse-feature test selected candidates only from the first four addition examples in each ABC form. The final four examples were confirmation data. Active controls were sampled from noncandidate features whose activation magnitude exceeded `1e-5` at the same layer and position. Confidence intervals are normal-approximation 95% intervals over intervention rows; paired contrasts use matching prompt/location keys. The raw CSV files permit bootstrap or hierarchical reanalysis without rerunning the model.

### Statistical and interpretive limits

The project contains multiple exploratory runs, so aggregate early effects are not treated as preregistered confirmatory evidence. The graph panel's same-form comparisons are confounded by template similarity. The path and head screens are small and in-sample. The held-out sparse-feature screen is the most stringent current test, but it still uses a synthetic arithmetic benchmark and a single sparse-transcoder model. We do not claim that null effects prove no reusable computation exists.

## Data and code availability

The public research release is available at https://github.com/josephletobar/formal-informal-reasoning/tree/v0.1.1-abc-publication. It includes the benchmark generator, derived raw CSV tables, figures, principal driver scripts, environment record, artifact hashes and verification checks. The serialized attribution graphs are retained in the authors' local archive rather than redistributed in the lightweight release; model weights and gated credentials are not redistributed.

## References

1. Anthropic. *Tracing the thoughts of a large language model*. 2025. https://www.anthropic.com/research/tracing-thoughts-language-model
2. Anthropic. *Open-sourcing circuit tracing tools*. 2025. https://www.anthropic.com/research/open-source-circuit-tracing
3. Anthropic. *Mapping the mind of a large language model*. 2024. https://www.anthropic.com/research/mapping-mind-language-model
4. Nature Machine Intelligence. *Content types and article format*. https://www.nature.com/natmachintell/content
