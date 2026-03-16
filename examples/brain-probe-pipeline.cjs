/**
 * Neural Preference Learning — Brain Probe Pipeline Example
 *
 * Demonstrates the template-based brain probe pipeline:
 * 1. Define personality templates (10-12 features each)
 * 2. Fire each template through the brain (read-only, no weight modification)
 * 3. Measure motor neuron response
 * 4. Compare trained vs untrained patterns
 * 5. Build context injection block
 *
 * Key insight: With 36 dimensions and 2000 sensory neurons, each dimension
 * gets ~55 neurons when all fire simultaneously — below the ~100-150 neuron
 * propagation threshold. Templates solve this by activating only 10-12
 * features at a time (~166-200 neurons each).
 *
 * This example uses mock data from a real production Agent Brain (20K neurons).
 */

const PROBE_TEMPLATES = {
  warm_devoted: {
    label: "Warm & Devoted", group: "companion",
    features: {
      emotional_warmth: 0.9, loyalty_expression: 0.8, empathy_depth: 0.8,
      supportiveness: 0.9, comfort_giving: 0.7, presence_awareness: 0.7,
      vulnerability: 0.5, intimacy_level: 0.6, memory_recall: 0.6,
      curiosity_about_user: 0.5, first_person_tone: 0.8, formality: 0.1
    }
  },
  playful_teasing: {
    label: "Playful & Teasing", group: "companion",
    features: {
      playfulness: 0.9, emotional_warmth: 0.6, humor_density: 0.7,
      intimacy_level: 0.5, curiosity_about_user: 0.7, vulnerability: 0.3,
      romantic_tone: 0.4, first_person_tone: 0.7, off_topic_tolerance: 0.6,
      emoji_usage: 0.4, formality: 0.0
    }
  },
  analytical: {
    label: "Analytical & Precise", group: "work",
    features: {
      response_length: 0.6, tool_count: 0.7, had_code: 0.8,
      had_data: 0.9, complexity: 0.8, technical_depth: 0.9,
      response_confidence: 0.7, explanation_depth: 0.8,
      was_proactive: 0.3, humor_density: 0.1, risk_appetite: 0.3,
      formality: 0.7
    }
  },
  casual: {
    label: "Casual & Friendly", group: "work",
    features: {
      humor_density: 0.7, first_person_tone: 0.8, cultural_flavor: 0.6,
      emoji_usage: 0.5, formality: 0.1, off_topic_tolerance: 0.5,
      response_confidence: 0.6, risk_appetite: 0.5, was_proactive: 0.6,
      question_count: 0.4
    }
  },
};

// Simulated probe results from a real trained Agent Brain (20K neurons, 542K synapses)
// After 50 iterations of warm_devoted training + 25 iterations of playful_teasing
const PRODUCTION_PROBE_RESULTS = {
  warm_devoted:     { avg_rate: 90.95, reinforce: 91.30, adjust: 90.25, explore: 91.30 },
  playful_teasing:  { avg_rate: 83.80, reinforce: 82.50, adjust: 84.10, explore: 84.80 },
  analytical:       { avg_rate: 89.90, reinforce: 88.37, adjust: 91.30, explore: 90.03 },
  casual:           { avg_rate: 70.10, reinforce: 67.07, adjust: 68.79, explore: 71.96 },
};

console.log("=== Brain Probe Pipeline Demo ===\n");
console.log("Templates:", Object.keys(PROBE_TEMPLATES).length);
console.log("Features per template:", Object.values(PROBE_TEMPLATES).map(t =>
  Object.keys(t.features).length
).join(", "));
console.log();

// Step 1: Calculate mean firing rate
const rates = Object.values(PRODUCTION_PROBE_RESULTS).map(r => r.avg_rate);
const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
console.log("Mean firing rate:", mean.toFixed(1), "Hz");
console.log();

// Step 2: Normalize and classify
console.log("=== Probe Results ===");
console.log("Template                    Rate(Hz)  Norm   Strength     Group");
console.log("─".repeat(70));

const results = {};
for (const [name, template] of Object.entries(PROBE_TEMPLATES)) {
  const probe = PRODUCTION_PROBE_RESULTS[name];
  const delta = probe.avg_rate - mean;
  const normalized = Math.max(0, Math.min(1, probe.avg_rate / (mean * 2)));

  let strength;
  if (delta > mean * 0.15) strength = "strong";
  else if (delta > mean * 0.05) strength = "moderate";
  else if (delta > mean * 0.01) strength = "slight";
  else if (delta < -mean * 0.1) strength = "suppressed";
  else if (delta < -mean * 0.03) strength = "weak";
  else strength = "neutral";

  results[name] = { ...template, avg_rate: probe.avg_rate, normalized, strength, delta };

  const arrow = delta > 0 ? "↑" : delta < 0 ? "↓" : "→";
  console.log(
    `${template.label.padEnd(28)} ${probe.avg_rate.toFixed(1).padStart(7)}  ${normalized.toFixed(2)}   ${(strength + " " + arrow).padEnd(12)} ${template.group}`
  );
}

// Step 3: Build context injection block
console.log("\n=== Context Injection Block ===");
const sorted = Object.entries(results)
  .filter(([_, t]) => t.strength !== "neutral")
  .sort((a, b) => b[1].avg_rate - a[1].avg_rate);

let ctx = "\n[Neural Pattern — live brain readout]\n";
const companion = sorted.filter(([_, t]) => t.group === "companion");
const work = sorted.filter(([_, t]) => t.group === "work");

if (companion.length > 0) {
  ctx += "Companion patterns: ";
  ctx += companion.map(([_, t]) => `${t.label}=${t.normalized.toFixed(2)} (${t.strength})`).join(", ") + "\n";
}
if (work.length > 0) {
  ctx += "Work patterns: ";
  ctx += work.map(([_, t]) => `${t.label}=${t.normalized.toFixed(2)} (${t.strength})`).join(", ") + "\n";
}
ctx += "Values 0-1: 0=untrained, 0.5=baseline, 1.0=heavily trained. Stronger patterns should be more prominent.\n";

console.log(ctx);

// Step 4: Show the neuron budget math
console.log("=== Neuron Budget Analysis ===");
console.log("Agent Brain: 2000 sensory neurons, 36 total dimensions");
console.log();
const budgetTable = [
  { dims: 1,  neurons: 2000, rate: 93.2,  works: "YES" },
  { dims: 6,  neurons: 333,  rate: 91.92, works: "YES" },
  { dims: 12, neurons: 166,  rate: 97.53, works: "YES — template sweet spot" },
  { dims: 36, neurons: 55,   rate: 0,     works: "NO — below propagation threshold" },
];
console.log("Active Dims  Neurons/Feature  Motor Response  Propagates?");
console.log("─".repeat(60));
for (const row of budgetTable) {
  console.log(
    `${String(row.dims).padStart(10)}  ${String(row.neurons).padStart(14)}  ${(row.rate + " Hz").padStart(12)}  ${row.works}`
  );
}
console.log();
console.log("Minimum propagation threshold: ~100-150 neurons per feature");
console.log("Template-based probing (10-12 features) stays well above this threshold.");
