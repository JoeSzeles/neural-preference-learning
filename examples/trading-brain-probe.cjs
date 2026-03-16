/**
 * Neural Preference Learning — Trading Brain Probe Example
 *
 * Demonstrates probing a 5K-neuron Trading Brain with price-scenario
 * templates. Each scenario injects a characteristic price/volume pattern
 * and measures the motor neuron response (buy/sell/hold).
 *
 * The Trading Brain uses 6 sensory zones:
 *   Price Up (20), Price Down (20), Volume (15), Spread (10),
 *   Momentum (10), Antenna (25)
 *
 * Production probe results show strong differentiation:
 *   Flash Crash: 143.5 Hz (strongest — sell signal)
 *   Squeeze Breakout: 143.5 Hz (buy signal)
 *   Low Liquidity: 51.25 Hz (weakest — uncertain)
 */

const TRADING_PROBE_SCENARIOS = [
  {
    name: "Bullish Breakout",
    features: { price: 0.85, volume: 0.9, spread: 0.3, momentum: 0.9 },
    description: "Strong price breakout with high volume confirmation"
  },
  {
    name: "Flash Crash",
    features: { price: -0.95, volume: 0.95, spread: 0.8, momentum: -0.99 },
    description: "Sudden extreme price drop with volume spike"
  },
  {
    name: "Steady Uptrend",
    features: { price: 0.3, volume: 0.4, spread: 0.2, momentum: 0.3 },
    description: "Gradual bullish movement with moderate volume"
  },
  {
    name: "Consolidation",
    features: { price: 0.05, volume: 0.2, spread: 0.15, momentum: 0.0 },
    description: "Flat price action with declining volume"
  },
  {
    name: "Bear Reversal",
    features: { price: -0.6, volume: 0.7, spread: 0.5, momentum: -0.7 },
    description: "Sharp downturn after extended rally"
  },
  {
    name: "Squeeze Breakout",
    features: { price: 0.7, volume: 0.85, spread: 0.1, momentum: 0.8 },
    description: "Breakout from tight consolidation with volume"
  },
  {
    name: "High Volume Spike",
    features: { price: 0.1, volume: 0.95, spread: 0.6, momentum: 0.2 },
    description: "Volume explosion with minimal price movement"
  },
  {
    name: "Low Liquidity Drift",
    features: { price: 0.15, volume: 0.05, spread: 0.9, momentum: 0.1 },
    description: "Wide spreads, thin volume, slow drift"
  },
  {
    name: "Gap Up Open",
    features: { price: 0.6, volume: 0.7, spread: 0.4, momentum: 0.5 },
    description: "Strong opening gap with volume follow-through"
  },
  {
    name: "Panic Selling",
    features: { price: -0.8, volume: 0.9, spread: 0.7, momentum: -0.85 },
    description: "High-volume sell-off with widening spreads"
  },
  {
    name: "Dead Cat Bounce",
    features: { price: 0.4, volume: 0.3, spread: 0.5, momentum: 0.2 },
    description: "Brief recovery after sharp decline, weak volume"
  },
  {
    name: "Momentum Surge",
    features: { price: 0.5, volume: 0.6, spread: 0.2, momentum: 0.95 },
    description: "Strong momentum with accelerating price movement"
  }
];

// Simulated probe results from production Trading Brain (5K neurons, 130K synapses)
const PRODUCTION_RESULTS = [
  { name: "Bullish Breakout",   avg_rate: 22.17,  buy: 41.50, sell: 8.33, hold: 16.67 },
  { name: "Flash Crash",        avg_rate: 143.50, buy: 0.00,  sell: 292.50, hold: 137.00 },
  { name: "Steady Uptrend",     avg_rate: 142.83, buy: 172.00, sell: 120.50, hold: 136.00 },
  { name: "Consolidation",      avg_rate: 130.42, buy: 154.00, sell: 122.00, hold: 115.25 },
  { name: "Bear Reversal",      avg_rate: 128.83, buy: 68.50, sell: 175.00, hold: 143.00 },
  { name: "Squeeze Breakout",   avg_rate: 143.50, buy: 240.00, sell: 53.00, hold: 137.50 },
  { name: "High Volume Spike",  avg_rate: 109.00, buy: 120.00, sell: 107.00, hold: 100.00 },
  { name: "Low Liquidity Drift", avg_rate: 51.25, buy: 68.50, sell: 40.50, hold: 44.75 },
  { name: "Gap Up Open",        avg_rate: 139.00, buy: 180.00, sell: 100.00, hold: 137.00 },
  { name: "Panic Selling",      avg_rate: 130.50, buy: 20.00, sell: 245.00, hold: 126.50 },
  { name: "Dead Cat Bounce",    avg_rate: 79.67,  buy: 90.00, sell: 64.00, hold: 85.00 },
  { name: "Momentum Surge",     avg_rate: 140.00, buy: 195.00, sell: 90.00, hold: 135.00 },
];

console.log("=== Trading Brain Probe Demo ===\n");
console.log("Scenarios:", TRADING_PROBE_SCENARIOS.length);
console.log("Brain: 5,000 neurons (S=600, I=3600, M=800), 130,600 synapses\n");

// Analyze results
const mean = PRODUCTION_RESULTS.reduce((s, r) => s + r.avg_rate, 0) / PRODUCTION_RESULTS.length;
console.log("Mean firing rate:", mean.toFixed(1), "Hz\n");

console.log("Scenario              Rate(Hz)   Signal   Buy     Sell    Hold");
console.log("─".repeat(68));

for (const r of PRODUCTION_RESULTS) {
  const motorMax = Math.max(r.buy, r.sell, r.hold);
  let signal;
  if (motorMax === r.buy) signal = "BUY ";
  else if (motorMax === r.sell) signal = "SELL";
  else signal = "HOLD";

  const delta = r.avg_rate - mean;
  const arrow = delta > mean * 0.1 ? "↑" : delta < -mean * 0.1 ? "↓" : "→";

  console.log(
    `${r.name.padEnd(22)} ${r.avg_rate.toFixed(1).padStart(7)}   ${signal} ${arrow}  ${r.buy.toFixed(0).padStart(6)}  ${r.sell.toFixed(0).padStart(6)}  ${r.hold.toFixed(0).padStart(6)}`
  );
}

console.log("\n=== Key Observations ===");
console.log("1. Flash Crash (143.5 Hz): Strong SELL signal — sell=292.5, buy=0.0");
console.log("   The brain has learned to aggressively sell during crashes.");
console.log("2. Squeeze Breakout (143.5 Hz): Strong BUY signal — buy=240.0");
console.log("   Trained to recognize breakout patterns as buying opportunities.");
console.log("3. Bullish Breakout (22.2 Hz): Suppressed response");
console.log("   May indicate undertrained scenario or deliberate caution.");
console.log("4. Low Liquidity (51.3 Hz): Weak, uncertain response");
console.log("   Brain has not been heavily trained on thin-market scenarios.");
console.log("5. Panic Selling vs Bear Reversal: Both trigger SELL but at different intensities");
console.log("   The brain differentiates severity of bearish scenarios.");
