# Neural Preference Learning: Real-Time Spiking Network Augmentation for Persistent LLM Agent Adaptation

**Joe Szeles**

*BrainJar / OpenClaw Project*

---

## Abstract

Large language model (LLM) agents are fundamentally stateless with respect to individual user preferences — each interaction begins with no persistent memory of what a particular user values in agent responses. Current approaches to preference alignment, such as reinforcement learning from human feedback (RLHF), operate as batch pre-deployment processes that produce a generic alignment rather than personal adaptation. This paper presents *Neural Preference Learning* (NPL), a novel architecture that augments LLM agents with companion spiking neural networks (SNNs) operating as persistent, real-time preference substrates. When users provide natural language feedback (e.g., "great answer" or "no, fix this"), the system classifies sentiment and maps the preceding agent response to a multi-dimensional feature vector, which is then stimulated through dedicated sensory neuron populations with sugar (positive) or pain (negative) reinforcement. Over time, Hebbian synaptic plasticity shapes each network into a persistent "intuition" model that survives across sessions, model swaps, and architecture changes.

The production system implements a dual-brain architecture: a 20,000-neuron Agent Brain (542,000 synapses) for personality and communication preference learning across 36 behavioral dimensions, and a 5,000-neuron Trading Brain (130,600 synapses) for market pattern recognition. A template-based neural probe pipeline enables real-time readout of trained synaptic patterns, injecting a live "neural fingerprint" into agent context to guide response generation.

**Keywords:** spiking neural network, LLM agent, preference learning, reinforcement, RLHF, LIF neuron, Hebbian learning, multi-agent system, neural probe, brain fingerprint

---

## 1. Introduction

The deployment of large language model agents in interactive, long-running contexts exposes a fundamental limitation: LLMs do not learn from individual user interactions. A user who consistently prefers concise, data-rich responses must re-state this preference in every session. The model's weights are frozen at deployment time, and while prompt engineering and system instructions can encode some preferences, these are brittle, manually maintained, and lack the adaptive quality of genuine learning.

Reinforcement Learning from Human Feedback (RLHF) [1, 2] has emerged as the dominant approach to aligning LLM outputs with human preferences. However, RLHF operates at training time with aggregated feedback from many evaluators, producing a generic preference alignment. It does not adapt to individual users, operates in batch rather than real-time, and requires expensive retraining cycles. Personal fine-tuning approaches face similar batch-processing constraints and risk catastrophic forgetting.

Memory-augmented architectures such as MemGPT [3] address persistence by storing interaction history as text, but text-based memory does not generalize — it recalls what happened, not what the user *prefers*. Cognitive architectures like ACT-R [4] and SOAR [5] model human cognition with symbolic production rules, but integrate poorly with neural language models.

This paper bridges these gaps by introducing *Neural Preference Learning* (NPL), which pairs an LLM agent with biologically-inspired spiking neural networks that operate as persistent, continuously-adapting preference substrates. The key contributions are:

1. **A hybrid architecture** combining symbolic LLM reasoning with sub-symbolic spiking network learning, where the SNN serves as a real-time preference memory that the LLM cannot modify or corrupt.
2. **Conversational reinforcement** — user feedback in natural language is automatically classified and translated to neurochemical-analog signals (sugar/pain) that modify synaptic weights through Hebbian plasticity.
3. **36-dimensional response encoding** — agent responses are mapped across 36 behavioral dimensions (content, behavior, style, personality, identity, companion) that activate dedicated sensory neuron populations.
4. **Template-based neural probing** — a pipeline for reading trained synaptic patterns by firing template stimulation patterns and measuring motor neuron response, producing a live "neural fingerprint" injected into agent context.
5. **Dual-brain architecture** — separate 20K-neuron Agent Brain (personality/communication) and 5K-neuron Trading Brain (market patterns) with independent weight persistence.
6. **Production deployment** — the system is implemented and running in a live multi-agent platform (OpenClaw), not as a simulation or theoretical framework.

---

## 2. Related Work

### 2.1 Reinforcement Learning from Human Feedback

Christiano et al. [1] introduced learning reward models from human preferences for training RL agents. Ouyang et al. [2] applied this framework to language models, producing InstructGPT. Constitutional AI [6] further automates preference learning through self-critique. All these approaches operate at training time and produce population-level alignment rather than personal adaptation.

### 2.2 Memory-Augmented LLM Agents

MemGPT (now Letta) [3] introduces virtual context management with persistent memory tiers. Retrieval-augmented generation (RAG) systems store and retrieve relevant context. These approaches maintain factual memory but do not perform adaptive learning — they recall, but do not generalize preferences from sparse feedback signals.

### 2.3 Cognitive Architectures

ACT-R [4] and SOAR [5] model human cognition through production rules and symbolic working memory. While they support learning through chunking and utility learning, they operate in a symbolic paradigm that integrates poorly with continuous neural network representations.

### 2.4 Spiking Neural Networks

Maass [7] established the theoretical foundations of spiking neural networks as the "third generation" of neural network models. Tavanaei et al. [8] surveyed deep learning in SNNs. The Drosophila connectome mapping project [9] provided detailed biological neural circuit data. SNNs have been applied to pattern recognition and temporal processing, but their application as companion preference substrates for LLM agents has not been previously explored.

### 2.5 Personal LLM Adaptation

LoRA [10] and QLoRA [11] enable efficient fine-tuning of LLMs. However, personal fine-tuning from conversational feedback requires accumulating sufficient training data, runs in batch mode, and risks destabilizing the base model. NPL sidesteps these issues by maintaining preference learning in a separate neural substrate that does not modify the LLM's weights.

---

## 3. Architecture

### 3.1 Dual-Brain System Overview

The production NPL system implements two independent spiking neural networks:

```
                       ┌──────────────────────────────────────┐
                       │         Agent Brain (20K neurons)    │
User ←→ LLM Agent     │  Sensory(2000) → Inter(14K) → Motor(4K)  │
         ↓ ↑          │  542,000 synapses · 36 dimensions    │
  Sentiment Detector   │  Motor: Reinforce / Adjust / Explore │
         ↓             │  Mushroom Body: 2,800 neurons        │
  Feature Vector(36-D) │                                      │
         ↓             └──────────────────────────────────────┘
  Sugar/Pain Feedback          ↓ Neural Fingerprint ↓
                        ┌─────────────────────┐
                        │  Context Injection   │
                        │  → LLM System Prompt │
                        └─────────────────────┘

                       ┌──────────────────────────────────────┐
Market ←→ Trading Bot  │       Trading Brain (5K neurons)     │
         ↓ ↑          │  Sensory(600) → Inter(3600) → Motor(800)  │
  Price/Volume Data    │  130,600 synapses · 6 zones          │
         ↓             │  Motor: Buy / Sell / Hold            │
  Pattern Feedback     │  Mushroom Body: 40 neurons           │
                       └──────────────────────────────────────┘
```

### 3.2 Agent Brain — Personality & Communication Learning

The Agent Brain is a 20,000-neuron LIF network dedicated to learning user preferences for agent personality, communication style, and response characteristics:

- **Sensory neurons** (S=2,000): Mapped to 36 behavioral dimensions via 6 sensory zones
- **Interneurons** (I=14,000): Process and associate patterns through recurrent connections, including a 2,800-neuron mushroom body for memory consolidation
- **Motor neurons** (M=4,000): Three-region output — Reinforce (strengthen current preferences), Adjust (modify/adapt), Explore (try new patterns)
- **Synapses**: 542,000 total connections

**Sensory zone allocation:**

| Zone | Start | Count | Dimensions |
|---|---|---|---|
| Content Features | 0 | 440 | response_length, tool_count, had_code, had_data, error_content, complexity, explanation_depth |
| Behavior Features | 440 | 360 | was_proactive, question_count, speed_completeness, off_topic_tolerance |
| Style Features | 800 | 360 | formality, list_usage, emoji_usage, visual_usage, first_person_tone |
| Personality Features | 1160 | 360 | risk_appetite, humor_density, technical_depth, response_confidence, cultural_flavor |
| Identity Features | 1520 | 200 | topic_hash, agent_id_hash |
| Meta Features | 1720 | 280 | emotional_warmth, intimacy_level, playfulness, loyalty_expression, memory_recall, empathy_depth, romantic_tone, vulnerability, presence_awareness, supportiveness, curiosity_about_user, comfort_giving, response_time |

### 3.3 Trading Brain — Market Pattern Recognition

The Trading Brain is a separate 5,000-neuron LIF network for financial market pattern learning:

- **Sensory neurons** (S=600): Mapped to price movement zones
- **Interneurons** (I=3,600): Pattern association
- **Motor neurons** (M=800): Buy/Sell/Hold output signals

**Sensory zone allocation:**

| Zone | Start | Count | Function |
|---|---|---|---|
| Price Up | 0 | 20 | Price increase detection |
| Price Down | 20 | 20 | Price decrease detection |
| Volume | 40 | 15 | Volume/trade activity |
| Spread | 55 | 10 | Spread width / liquidity |
| Momentum | 65 | 10 | Price momentum / acceleration |
| Antenna | 75 | 25 | Pressure sensing (volume spikes, rapid moves, flash crashes) |

### 3.4 LIF Neuron Model

Each neuron follows Leaky Integrate-and-Fire dynamics:

```
V(t+1) = V(t) × decay + Σ(w_ij × spike_j) + I_ext
if V(t+1) > V_threshold: spike, V → V_reset
```

**Agent Brain parameters:** `w_syn = 12.0`, `r_poi = 150`, `tau_syn = 5`
**Trading Brain parameters:** `w_syn = 12.0`, `r_poi = 150`, `tau_syn = 5`

### 3.5 Persistence Layer

All feedback interactions are stored in a PostgreSQL database with a local JSON file mirror:

- **Primary storage**: `neural_feedback` table with composite unique index (timestamp, agent_id, session_id, sentiment)
- **File mirror**: JSON file written on every DB write for portability without a database
- **Daily backups**: Rotated 30-day snapshots
- **Engram backups**: On-demand brain state snapshots for rollback
- **Startup sync**: Bidirectional merge on boot, with composite key deduplication and ISO timestamp normalization

Synaptic weights are serialized to binary JSON (`brain-weights.json`) and restored on boot. If the network architecture changes, stored weights are discarded and interactions are replayed from persistent storage.

---

## 4. Methodology

### 4.1 36-Dimensional Response Feature Encoding

When an agent produces a response, the system extracts a 36-dimensional feature vector across six categories:

**Content (7 dimensions):**

| Feature | Range | Description |
|---|---|---|
| `response_length` | [0, 1] | Character count normalized by 2,000 |
| `tool_count` | [0, N] | Number of tools invoked |
| `had_code` | {0, 1} | Whether the response contained code blocks |
| `had_data` | {0, 1} | Whether the response referenced data/tables/numbers |
| `had_error` | {0, 1} | Whether the response mentions errors or failures |
| `complexity` | [0, 1] | Composite measure of code, tools, and data |
| `explanation_depth` | [0, 1] | How much the response explains (relative to length) |

**Behavior (4 dimensions):** `was_proactive`, `question_count`, `speed_completeness`, `off_topic_tolerance`

**Style (5 dimensions):** `formality`, `list_usage`, `emoji_usage`, `visual_usage`, `first_person_tone`

**Personality (5 dimensions):** `risk_appetite`, `humor_density`, `technical_depth`, `response_confidence`, `cultural_flavor`

**Identity (2 dimensions):** `topic_hash`, `agent_id_hash`

**Companion/Interpersonal (12 dimensions):** `emotional_warmth`, `intimacy_level`, `playfulness`, `loyalty_expression`, `memory_recall`, `empathy_depth`, `romantic_tone`, `vulnerability`, `presence_awareness`, `supportiveness`, `curiosity_about_user`, `comfort_giving`

**Performance (1 dimension):** `response_time`

Each feature activates its mapped sensory neuron population using population coding: for `k` neurons allocated to a feature, the `i`-th neuron receives stimulation proportional to `feature_value × gaussian(i, mean=k/2, sigma=k/4)`.

### 4.2 Neuron Budget and Signal Propagation Threshold

A critical design constraint discovered during production deployment: the number of sensory neurons allocated to each feature dimension must exceed a minimum threshold (~100-150 neurons) for the signal to propagate through the interneuron network and produce measurable motor output.

With the Agent Brain (2,000 sensory neurons, 36 dimensions), each dimension receives ~55 neurons when all dimensions fire simultaneously. This is below the propagation threshold, resulting in zero motor output. The solution: **template-based stimulation**, where only 10-12 relevant dimensions are activated per stimulation event, giving each active dimension ~166-200 neurons — well above the threshold.

| Active Dimensions | Neurons/Feature | Motor Response |
|---|---|---|
| 1 | 2,000 | 93 Hz (full propagation) |
| 2 | 1,000 | 100 Hz |
| 6 | 333 | 92 Hz |
| 12 | 166 | 98 Hz |
| 36 | 55 | 0 Hz (below threshold) |

This finding has direct implications for SNN architecture design: the sensory neuron budget divided by the maximum number of simultaneously active features must remain above the minimum propagation threshold.

### 4.3 Training Templates

Rather than stimulating all 36 dimensions at once, training uses predefined personality templates that activate 10-12 related dimensions with characteristic intensity values:

**Companion templates:**
- **Warm & Devoted**: emotional_warmth=0.9, loyalty=0.8, empathy=0.8, supportiveness=0.9, comfort=0.7
- **Playful & Teasing**: playfulness=0.9, humor=0.7, curiosity=0.7, intimacy=0.5, warmth=0.6
- **Empathetic & Deep**: empathy=0.9, vulnerability=0.8, warmth=0.8, intimacy=0.7
- **Romantic & Poetic**: romantic=0.9, warmth=0.8, vulnerability=0.7, intimacy=0.8
- **Protective & Loyal**: loyalty=0.9, support=0.9, comfort=0.8, confidence=0.8
- **Curious & Engaged**: curiosity=0.9, presence=0.8, memory=0.8, empathy=0.6

**Work templates:**
- **Analytical & Precise**: code=0.8, data=0.9, complexity=0.8, technical=0.9
- **Creative & Bold**: risk=0.9, humor=0.6, proactive=0.8, confidence=0.8
- **Patient & Thorough**: length=0.9, depth=0.9, lists=0.7, completeness=0.9
- **Concise & Direct**: length=0.2, confidence=0.9, formality=0.6
- **Casual & Friendly**: humor=0.7, first_person=0.8, emoji=0.5, cultural=0.6
- **Cautious & Safe**: risk=0.1, formality=0.8, depth=0.7, questions=0.6

Each template is trained with sugar feedback and slight jitter (±0.05) to build robust synaptic pathways.

### 4.4 Sentiment Classification

User feedback is classified through keyword matching:

- **Positive** (sugar trigger): "good", "great", "perfect", "yes", "nice", "excellent", "love", "awesome", "correct", "exactly", "thanks", "helpful", "works", "right", "amazing", "fantastic", "wonderful", "brilliant", "superb"
- **Negative** (pain trigger): "no", "wrong", "bad", "redo", "fix", "broken", "terrible", "useless", "stop", "hate", "awful", "horrible", "worse", "ugly", "stupid", "fail", "error", "bug", "crash", "mess"
- **Neutral**: No sentiment keywords detected (no reinforcement applied)

### 4.5 Reinforcement Protocol

When a user message triggers sentiment detection:

1. The feature vector of the *previous* agent response is retrieved
2. The feature vector is stimulated through the preference zone neurons (5 simulation steps)
3. Sugar (positive) or pain (negative) feedback is applied:
   - **Sugar**: All recently-active synapses in the stimulated pathway have their weights increased by `Δw = η × pre_spike × post_spike` (Hebbian)
   - **Pain**: Active synapses have weights decreased by `Δw = -η × pre_spike × post_spike` (anti-Hebbian)
4. The complete interaction record (timestamp, agent ID, feature vector, sentiment, brain response, raw text char count) is persisted to the database and file mirror

### 4.6 Replay Mechanism

To survive architecture changes (network resizing), the system implements two replay mechanisms:

- **Trading replay**: Stored per-instrument tick patterns replayed through `stimulateFromPrice()` with original feedback
- **Preference replay**: Last 200 preference interactions replayed from the database through `stimulateFromPreference()` with original sentiment-based reinforcement

Both replays execute automatically on boot when the system detects an architecture mismatch between saved weights and current configuration.

---

## 5. Brain Probe Pipeline

### 5.1 The Probe Problem

A trained spiking neural network stores information in its synaptic weights, but reading those weights directly does not reveal what patterns the network has learned. The weight matrix is high-dimensional (542,000 synapses for the Agent Brain) and the relationship between individual weights and learned behaviors is non-linear.

The probe must answer: "What behavioral patterns has this brain been trained on, and how strongly?"

### 5.2 Template-Based Probing

The probe fires the same templates used for training through the network and measures the motor neuron response. Higher firing rates indicate stronger learned associations:

```
For each template T in PROBE_TEMPLATES:
  1. Construct feature vector from T.features (10-12 active dimensions)
  2. POST to brain engine /stimulate-preference (no feedback, read-only)
  3. Read avg_rate, reinforce/adjust/explore signals
  4. Store result

Compute mean firing rate across all templates
Normalize each template: normalized = avg_rate / (mean × 2)
Assign strength labels: strong (>15% above mean), moderate, slight, neutral, weak, suppressed
```

**Production probe results (Agent Brain, after training):**

| Template | Avg Rate (Hz) | Normalized | Strength | Dominant Signal |
|---|---|---|---|---|
| Warm & Devoted | 90.95 | 0.53 | moderate | Reinforce |
| Romantic & Poetic | 90.17 | 0.53 | moderate | Reinforce |
| Curious & Engaged | 86.83 | 0.51 | slight | Explore |
| Casual & Friendly | 70.10 | 0.41 | suppressed | - |

**Production probe results (Trading Brain):**

| Scenario | Avg Rate (Hz) | Dominant Motor | Character |
|---|---|---|---|
| Flash Crash | 143.50 | SELL | Strong response to crash patterns |
| Squeeze Breakout | 143.50 | BUY | Strong response to breakout patterns |
| Steady Uptrend | 142.83 | BUY | Trained on gradual bullish moves |
| Low Liquidity | 51.25 | BUY | Weak response to thin markets |

### 5.3 Neural Fingerprint Context Injection

The probe results are formatted into a compact text block and injected into the LLM's system prompt:

```
[Neural Pattern — live brain readout]
Companion patterns: Warm & Devoted=0.53 (moderate), Curious & Engaged=0.51 (slight)
Work patterns: Analytical & Precise=0.56 (moderate), Creative & Bold=0.53 (moderate)
Values 0-1: 0=untrained, 0.5=baseline, 1.0=heavily trained. Stronger patterns should be more prominent.
```

A companion interpretation document (BRAIN_PATTERNS.md) is provided in the agent's knowledge base, enabling the LLM to translate probe values into behavioral adjustments.

### 5.4 Injection Gate

To prevent injection of noise before sufficient training data exists, the system implements a stimulation gate:

- Preference context injection is only activated after 3+ real brain stimulations per session
- This prevents injecting stale or random data after server restarts
- The gate resets on each server restart, requiring fresh stimulation to re-enable injection

### 5.5 Cache Management

Probe results are cached for 45 seconds to avoid excessive brain stimulation. The cache is per-brain (Agent Brain and Trading Brain have independent caches).

---

## 6. Implementation

### 6.1 Runtime Environment

The system is implemented in Node.js and deployed as part of the OpenClaw multi-agent platform. Each brain engine runs as an internal HTTP server (`brain-engine-server.cjs`), auto-spawned by the platform's process manager. All inter-component communication uses HTTP REST APIs authenticated with API keys and session tokens.

### 6.2 Network Simulation

The LIF network simulation runs synchronously in JavaScript. At each simulation step:

1. External input currents are injected into sensory neurons
2. Membrane potentials are updated with decay and synaptic input
3. Neurons exceeding threshold fire and reset
4. Synaptic weights are updated based on spike timing
5. Motor neuron firing rates are accumulated

Performance: A 20,000-neuron network completes 10 simulation steps in ~15ms on commodity hardware.

### 6.3 API Surface

**Agent Brain endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/api/neural-feedback/brain-probe` | GET | Template-based agent brain probe |
| `/api/neural-feedback/injection-preview` | GET | Preview current injection context |
| `/api/neural-feedback/status` | GET | Feedback statistics |
| `/api/neural-feedback/history` | GET | Recent interaction records |
| `/api/agent-brain/train-template` | POST | Train with personality template |
| `/api/agent-brain/status` | GET | Agent brain status/architecture |

**Trading Brain endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/api/brain/probe-trading` | GET | Scenario-based trading brain probe |
| `/api/brain/status` | GET | Trading brain status |
| `/api/brain/stimulate-price` | POST | Stimulate with price data |

---

## 7. Evaluation Framework

### 7.1 Preference Convergence

The primary metric is the ratio of positive to negative feedback over time. If the system is learning effectively, this ratio should increase as the agent's behavior is guided by accumulated preference data. Measurement: sliding window of last 50 interactions, tracked per agent.

### 7.2 Probe Differentiation

Template-based probing provides a quantitative measure of learning: the standard deviation of firing rates across templates. An untrained brain shows uniform rates (~80-100 Hz); a trained brain shows differentiation (trained templates fire 10-30% above mean, untrained templates fire below). Higher standard deviation = more learning.

### 7.3 Cross-Session Persistence

The system's ability to maintain learned preferences across restarts, model swaps, and architecture changes is tested by:

1. Training with a set of interactions
2. Restarting the system (verifying weight restoration and probe consistency)
3. Swapping the LLM model (verifying brain independence)
4. Changing network architecture (verifying replay reconstruction)

### 7.4 Model Independence

A critical property: preferences persist across LLM model changes. The same brain network can serve GPT-4, Claude, Grok, or any other model without retraining — the preference substrate operates at the feature vector level, not the model weight level.

---

## 8. Discussion

### 8.1 Novelty

NPL represents a fundamentally different approach to LLM preference alignment:

| Aspect | RLHF | MemGPT | NPL |
|---|---|---|---|
| Timing | Pre-deployment | Runtime (text) | Runtime (neural) |
| Personalization | Population | Per-user (text) | Per-user (synaptic) |
| Learning type | Batch gradient | None (retrieval) | Online Hebbian |
| Persistence | Model weights | Text database | Synaptic weights |
| Model-agnostic | No | Partially | Yes |
| Survives model swap | No | Yes (text) | Yes (neural) |
| Readable fingerprint | No | Partial (text dump) | Yes (probe pipeline) |
| Dual-domain | No | No | Yes (personality + trading) |

### 8.2 Biological Inspiration

The architecture draws directly from the Drosophila melanogaster olfactory learning circuit, where mushroom body Kenyon cells form associative memories between sensory inputs and reward/punishment signals. The mapping is:

- Sensory neurons ↔ Olfactory receptor neurons
- Preference zone ↔ Antennal lobe projection neurons
- Interneurons ↔ Kenyon cells
- Sugar/pain feedback ↔ Dopaminergic reward neurons
- Motor outputs ↔ Mushroom body output neurons

### 8.3 The Neuron Budget Constraint

The discovery that signal propagation requires a minimum neuron-per-feature threshold (~100-150 neurons) has implications for SNN architecture design. This constraint is analogous to biological neural circuits where population coding requires sufficient neuron counts for reliable signal transmission. Template-based stimulation (activating 10-12 features simultaneously rather than all 36) is the practical solution — and is also more biologically plausible, as real neural systems process stimuli in context-specific patterns rather than uniform activation.

### 8.4 Privacy

All preference learning occurs locally. No interaction data leaves the system. The neural substrate operates on abstract feature vectors, not raw text — raw text is never injected into context, only character counts are logged. Even the synaptic weights themselves encode no readable information about the user's preferences without the probe pipeline to interpret them.

### 8.5 Limitations

1. **Stochastic variation**: Spiking neural networks produce inherently variable probe results due to Poisson-process spike generation. Relative ordering between templates is meaningful but absolute values fluctuate ±10-15% between probes.
2. **Sentiment detection**: Keyword-based classification produces false positives/negatives and misses nuanced feedback.
3. **No convergence guarantee**: Hebbian learning in recurrent networks is not guaranteed to converge to optimal preference representation.
4. **Training template bias**: The predefined templates impose a structure on the personality space that may not capture all possible user preferences.
5. **Evaluation**: The system is deployed but lacks controlled experimental evaluation against baselines.

---

## 9. Future Work

### 9.1 Pre-Response Quality Gating

The brain could preview a planned response's feature vector before generation, providing a preference quality estimate that guides response strategy selection.

### 9.2 Multi-User Support

Separate brain instances per user would enable personal preference models in multi-user deployments, with an optional "consensus brain" aggregating cross-user patterns.

### 9.3 GPU-Accelerated Networks

Larger networks (100K+ neurons) with GPU-accelerated simulation would increase representational capacity and enable more sophisticated temporal pattern learning.

### 9.4 Neural-to-LoRA Bridge

Accumulated preference data from the neural substrate could periodically generate LoRA fine-tuning datasets, creating a pipeline from real-time neural reinforcement to model weight adaptation.

### 9.5 NLP Sentiment Analysis

Replacing keyword-based sentiment with a dedicated small language model would improve classification accuracy, particularly for nuanced, implicit, or sarcastic feedback.

### 9.6 Attention-Guided Feature Extraction

Learning which response features matter most (attention over the feature vector) would enable the system to adaptively weight feature dimensions based on accumulated evidence.

### 9.7 Cross-Brain Correlation

Correlating Agent Brain personality probes with Trading Brain market response patterns could reveal relationships between operator personality preferences and trading behavior.

---

## 10. Conclusion

Neural Preference Learning introduces a new paradigm for LLM agent adaptation: companion spiking neural networks that perform real-time, personal, persistent preference learning from natural language feedback. Unlike RLHF, it operates at runtime with individual users. Unlike text-based memory, it generalizes from sparse signals through synaptic plasticity. Unlike personal fine-tuning, it preserves the base model's capabilities while maintaining an independent preference substrate that survives model swaps and architecture changes.

The production system implements a dual-brain architecture — a 20,000-neuron Agent Brain for personality and communication preferences across 36 behavioral dimensions, and a 5,000-neuron Trading Brain for market pattern recognition — with template-based probing for live readout of learned patterns. The probe pipeline enables a continuous feedback loop: train the brain, probe its learned patterns, inject those patterns into agent context, observe the resulting behavior, and receive user feedback to further refine the brain.

The core insight is that LLM agents benefit from a separate neural substrate for preference learning — one that operates on a different timescale (real-time vs. training epochs), at a different granularity (individual vs. population), and with a different learning rule (Hebbian plasticity vs. gradient descent). This hybrid architecture suggests that the future of personalized AI may not lie in modifying the LLM itself, but in augmenting it with complementary neural systems purpose-built for the learning tasks that LLMs cannot perform.

---

## References

[1] P. F. Christiano, J. Leike, T. Brown, M. Milani, S. Gilmer, and D. Amodei, "Deep reinforcement learning from human preferences," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[2] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, et al., "Training language models to follow instructions with human feedback," in *Advances in Neural Information Processing Systems*, vol. 35, 2022.

[3] C. Packer, S. Wooders, K. Lin, V. Fang, S. G. Patil, I. Stoica, and J. E. Gonzalez, "MemGPT: Towards LLMs as operating systems," *arXiv preprint arXiv:2310.08560*, 2023.

[4] J. R. Anderson, *How Can the Human Mind Occur in the Physical Universe?* Oxford University Press, 2007.

[5] J. E. Laird, *The Soar Cognitive Architecture*. MIT Press, 2012.

[6] Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, et al., "Constitutional AI: Harmlessness from AI feedback," *arXiv preprint arXiv:2212.08073*, 2022.

[7] W. Maass, "Networks of spiking neurons: The third generation of neural network models," *Neural Networks*, vol. 10, no. 9, pp. 1659-1671, 1997.

[8] A. Tavanaei, M. Ghodrati, S. R. Kheradpisheh, T. Masquelier, and A. Maida, "Deep learning in spiking neural networks," *Neural Networks*, vol. 111, pp. 47-63, 2019.

[9] S. Dorkenwald, A. Matsliah, A. R. Sterling, P. Schlegel, S. Yu, C. E. McKellar, et al., "Neuronal wiring diagram of an adult brain," *Nature*, vol. 634, pp. 124-138, 2024.

[10] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," in *International Conference on Learning Representations*, 2022.

[11] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs," in *Advances in Neural Information Processing Systems*, vol. 36, 2023.

---

*This paper describes a system that is implemented and deployed in production at https://openclaw-mechanicus.replit.app as part of the OpenClaw multi-agent platform with BrainJar neural engine integration. Source code and reference implementation are available at https://github.com/JoeSzeles/neural-preference-learning. Standalone brain engine installer at https://github.com/JoeSzeles/ClawBrain.*
