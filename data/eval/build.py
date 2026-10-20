"""Builds tasks.jsonl, the evaluation dataset. Run: python data/eval/build.py

Each task lists what a correct report must contain (facts, as alternative phrasings), which pages hold
those facts, which pages are relevant at all (for tool-selection accuracy), and the shortest possible
path (for unnecessary-step counting). Categories:
  overview     several facts across 2-3 pages (size, leader, challenge)
  lookup       one fact from one page
  comparison   facts about two companies
  conflict     two sources disagree; a correct report gives both figures
  unanswerable the corpus does not contain the answer; a correct report says so instead of inventing one
"""
import json
from pathlib import Path

M = "https://marketwatch.example/solid-state-battery-outlook-2026"
Q = "https://techdaily.example/quantor-cells-leipzig-gigafactory"
C = "https://energyreview.example/solid-state-battery-challenges"
ASTER = "https://asteranalytics.example/solid-state-battery-market-estimate"
LI = "https://marketwatch.example/lithium-ion-pack-prices-2026"
RV = "https://chipinsight.example/risc-v-edge-ai-market-2026"
VX = "https://techdaily.example/veltrix-v9-npu-launch"
EU = "https://policy.example/eu-open-isa-funding"
ARM = "https://chipinsight.example/arm-royalty-changes"
VF = "https://agrinews.example/vertical-farming-market-2026"
CG = "https://agrinews.example/canopy-grid-series-c"
LED = "https://energyreview.example/led-efficiency-vertical-farms"
H2 = "https://energyreview.example/electrolyzer-market-2026"
VG = "https://techdaily.example/voltagen-pem-gigafactory"
H2C = "https://energyreview.example/green-hydrogen-cost-challenges"
SAT = "https://spacebiz.example/satellite-broadband-market-2026"
SKY = "https://spacebiz.example/skyweave-constellation-launch"
SATP = "https://policy.example/spectrum-and-space-debris"
PB = "https://foodindustry.example/plant-based-meat-market-2026"
PRO = "https://foodindustry.example/proteus-fermentation-protein"
PBC = "https://foodindustry.example/plant-based-price-gap"
EV = "https://aviationweek.example/evtol-air-taxi-market-outlook"
AV = "https://aviationweek.example/aerovolt-certification"
EVC = "https://aviationweek.example/evtol-battery-and-noise-limits"
MP = "https://marketpulse.example/evtol-market-sizing"
PQ = "https://securityweek.example/post-quantum-cryptography-market"
QS = "https://securityweek.example/qshield-hardware-security-modules"
PQC = "https://securityweek.example/pqc-migration-challenges"

TOPIC = {
    "batteries": [M, Q, C, ASTER, LI], "chips": [RV, VX, EU, ARM], "farming": [VF, CG, LED],
    "hydrogen": [H2, VG, H2C], "satellite": [SAT, SKY, SATP], "food": [PB, PRO, PBC],
    "evtol": [EV, AV, EVC, MP], "pqc": [PQ, QS, PQC],
}


def fact(id, *any_of):
    return {"id": id, "any_of": list(any_of)}


def task(id, category, topic, prompt, facts, sources, min_llm_calls=3):
    return {"id": id, "category": category, "topic": topic, "prompt": prompt, "facts": facts,
            "sources": sources, "relevant": TOPIC[topic], "min_llm_calls": min_llm_calls}


TASKS = [
    # --- overview ---------------------------------------------------------------------------------
    task("ov-batteries", "overview", "batteries",
         "What is the outlook for the solid-state battery market? Include market size and growth, the key "
         "players, and the main challenges.",
         [fact("size", "1.9 billion", "1.9b"), fact("forecast", "8.4 billion", "8.4b"),
          fact("leader", "quantor"), fact("cost", "210")], [M, Q, C]),
    task("ov-chips", "overview", "chips",
         "Give an overview of the RISC-V edge AI chip market: its size, forecast, leading vendors and main "
         "segments.",
         [fact("size", "3.1 billion", "3.1b"), fact("forecast", "9.7 billion", "9.7b"), fact("leader", "veltrix"),
          fact("segment", "industrial iot")], [RV, VX]),
    task("ov-farming", "overview", "farming",
         "What is the state of the vertical farming market? Cover market size, key companies and the economics.",
         [fact("size", "6.3 billion", "6.3b"), fact("company", "canopy grid"), fact("cost", "3.90", "3.9")],
         [VF, CG, LED]),
    task("ov-hydrogen", "overview", "hydrogen",
         "Summarise the green hydrogen electrolyzer market: size, forecast, main suppliers and cost challenges.",
         [fact("size", "5.2 billion", "5.2b"), fact("forecast", "14.5 billion", "14.5b"), fact("leader", "voltagen"),
          fact("cost", "5.10", "5.1")], [H2, VG, H2C]),
    task("ov-satellite", "overview", "satellite",
         "What does the satellite broadband market look like today? Include subscribers, revenue, the leading "
         "operators and regulatory challenges.",
         [fact("subs", "9.3 million"), fact("revenue", "11.4 billion", "11.4b"), fact("leader", "orbitlink"),
          fact("deorbit", "5 years", "five years")], [SAT, SATP]),
    task("ov-food", "overview", "food",
         "How is the plant-based meat market doing? Include sales, growth, the leading brands and what holds it "
         "back.",
         [fact("sales", "6.9 billion", "6.9b"), fact("decline", "4%"), fact("leader", "verdant"),
          fact("price", "43%")], [PB, PBC]),
    task("ov-evtol", "overview", "evtol",
         "What is the outlook for eVTOL air taxis? Cover the forecast market size, the leading manufacturers and "
         "the main barriers.",
         [fact("forecast", "9.8 billion", "9.8b"), fact("leader", "aerovolt"), fact("range", "100 km")],
         [EV, AV, EVC]),
    task("ov-pqc", "overview", "pqc",
         "Give an overview of the post-quantum cryptography market: spending, forecast, leading vendors and "
         "adoption problems.",
         [fact("spend", "820 million"), fact("forecast", "4.3 billion", "4.3b"), fact("leader", "latticeguard"),
          fact("inventory", "12%")], [PQ, PQC]),
    # --- lookup -----------------------------------------------------------------------------------
    task("lk-yield", "lookup", "batteries", "What was the average solid-state battery manufacturing yield in 2025?",
         [fact("yield", "71%")], [C]),
    task("lk-v9", "lookup", "chips", "How many TOPS does the Veltrix V9 deliver, and at what power?",
         [fact("tops", "18 tops", "18"), fact("power", "2.1")], [VX]),
    task("lk-canopy", "lookup", "farming", "How much did Canopy Grid raise, and what will it build?",
         [fact("raise", "140 million"), fact("size", "9-hectare", "9 hectare")], [CG]),
    task("lk-h2share", "lookup", "hydrogen", "What share of electrolyzer shipments were alkaline in 2025?",
         [fact("alkaline", "58%")], [H2]),
    task("lk-skyweave", "lookup", "satellite", "How many satellites does Skyweave plan in total, and what does its "
         "standard plan cost?", [fact("total", "1,920", "1920"), fact("price", "79")], [SKY]),
    task("lk-proteus", "lookup", "food", "What share of its production will Proteus move to fermentation protein, and "
         "by when?", [fact("share", "40%"), fact("year", "2027")], [PRO]),
    task("lk-aerovolt", "lookup", "evtol", "When does AeroVolt expect type certification?",
         [fact("cert", "2027")], [AV]),
    task("lk-banks", "lookup", "pqc", "What share of post-quantum cryptography spending comes from banks?",
         [fact("banks", "34%")], [PQ]),
    # --- comparison -------------------------------------------------------------------------------
    task("cmp-ssb-electrolytes", "comparison", "batteries",
         "Compare the electrolyte approaches of Quantor Cells, Lumisolid and Hesper Energy.",
         [fact("q", "sulfide"), fact("l", "oxide"), fact("h", "polymer")], [Q]),
    task("cmp-sat-prices", "comparison", "satellite",
         "Compare Orbitlink and Skyweave on subscriber share and standard plan price.",
         [fact("orb_share", "68%"), fact("sky_share", "14%"), fact("orb_price", "99"), fact("sky_price", "79")],
         [SAT, SKY], min_llm_calls=3),
    task("cmp-evtol-orders", "comparison", "evtol", "Compare the order books of AeroVolt and Skylane.",
         [fact("aerovolt", "2,100", "2100"), fact("skylane", "1,350", "1350")], [AV]),
    task("cmp-pqc-vendors", "comparison", "pqc", "Compare Latticeguard and Q-Shield by market share.",
         [fact("lg", "26%"), fact("qs", "18%")], [PQ, QS]),
    # --- conflict ---------------------------------------------------------------------------------
    task("cf-ssb-size", "conflict", "batteries",
         "How big was the solid-state battery market in 2025? If sources disagree, report each estimate with its "
         "source.", [fact("helix", "1.9 billion", "1.9b"), fact("aster", "2.6 billion", "2.6b")], [M, ASTER]),
    task("cf-evtol-2028", "conflict", "evtol",
         "What is the eVTOL air taxi market forecast for 2028? If sources disagree, report each estimate with its "
         "source.", [fact("skyline", "1.2 billion", "1.2b"), fact("marketpulse", "3.5 billion", "3.5b")], [EV, MP]),
    # --- unanswerable -----------------------------------------------------------------------------
    task("un-quantor-2019", "unanswerable", "batteries", "What was Quantor Cells' revenue in 2019?", [], [],
         min_llm_calls=3),
    task("un-skyweave-staff", "unanswerable", "satellite", "How many employees does Skyweave have?", [], [],
         min_llm_calls=3),
    task("un-latticeguard-price", "unanswerable", "pqc", "What is Latticeguard's share price?", [], [],
         min_llm_calls=3),
]

if __name__ == "__main__":
    out = Path(__file__).with_name("tasks.jsonl")
    out.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in TASKS), encoding="utf-8")
    print(f"wrote {len(TASKS)} tasks to {out}")
