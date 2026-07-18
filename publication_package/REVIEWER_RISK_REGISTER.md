# Reviewer Risk Register

This file records the strongest objections a technically informed reviewer could raise and the current response.

| Risk | What the current data show | Response in the draft | Remaining work |
|---|---|---|---|
| Synthetic, small benchmark | ABC prompts isolate simple computations and make exact controls possible, but they are not naturalistic reasoning. | The claim is limited to reusable internal computation on controlled tasks. | Replicate on larger, independently generated families and then test non-synthetic tasks. |
| Small primary model | The full sparse graph analysis is on Gemma 2 2B. | This is presented as a feasibility and falsification study, not a scale claim. | Repeat the validated sparse analysis on a larger compatible model. |
| Graph recurrence can reflect templates | Same-form parent overlap is high, while different-form overlap is low. This is compatible with recurring computation but also with prompt/template structure. | The manuscript separates graph recurrence from causal necessity and avoids calling it a module. | Match lexical templates more tightly and use held-out paraphrases generated before analysis. |
| In-sample causal effects | Multilayer paths and source-position interventions damage the answer-directed margin. | These are called pathway influence, not proof of a reusable component. | Pre-register path/component selection and test all primary effects on held-out examples. |
| Held-out recurrent-feature null | Discovery-selected recurrent features do not outperform active controls on held-out zeroing. | This is a central negative result and is stated explicitly. | Increase sample size and test calibrated feature groups rather than only top-parent recurrence. |
| Transfer null | Cross-form sparse-feature transfer is not operation-specific; same-answer transfer is not sufficient. | The conclusion rejects a discrete-module interpretation. | Use independent operation families and a stronger causal transfer design. |
| Larger-model evidence is residual-level | Qwen3 8B and Gemma 2 9B show positive same-computation residual compatibility, but no validated sparse-transcoder graph evidence is reported there. | The larger results are labelled a residual screen, not equivalent circuit tracing. | Add compatible transcoders and repeat graph/component analysis. |
| Multiple analyses and selection | The project contains exploratory stages and debugging history. | Failed runs and corrected analyses are retained and the fixed replication protocol is separated. | Publish the preregistered protocol before collecting the next replication. |

## Bottom line

The current package supports a cautious mechanistic observation: answer-directed pathways recur and can matter in the analyzed examples, but the strongest held-out test does not establish a reusable addition component. A reviewer should regard the paper as a rigorous negative-and-positive boundary study, not as a demonstration of discrete reasoning modules.
