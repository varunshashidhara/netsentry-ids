# NetSentry — AI-Based Network Intrusion Detection System

**Live demo:** [netsentry-ids.onrender.com](https://netsentry-ids.onrender.com) *(free-tier hosting — first load after inactivity may take 30-60 seconds to wake up. Supports Simulate and manual input; real packet capture requires running locally, see that section below for why.)*

A working prototype matching the architecture:

```
Network Logs → Python → ML Model → Flask API → SQL DB → Dashboard
```

## What's included

| File | Purpose |
|---|---|
| `data/KDDTrain.txt`, `data/KDDTest.txt` | Real NSL-KDD intrusion-detection benchmark dataset (125,973 train / 22,544 test rows, 41 traffic features each, labeled normal vs. DoS/Probe/R2L/U2R attacks) |
| `train_model.py` | Preprocesses the dataset and trains 3 models with scikit-learn |
| `app.py` | Flask API: real-time classification, SQLite alert storage, dashboard route |
| `templates/dashboard.html` | Live SOC-style console (auto-refreshing alert feed, stats, manual + simulated traffic injection) |
| `models/` | Saved trained models + encoders/scaler (created by `train_model.py`) |
| `ids_alerts.db` | SQLite database of alerts (created on first run) |

## Models trained

- **Random Forest** — binary classifier (normal vs. attack) **and** a second one for attack category (DoS / Probe / R2L / U2R)
- **Decision Tree** — binary classifier, used as a second opinion / sanity check against the Random Forest
- **Isolation Forest** — unsupervised anomaly detector trained only on normal traffic, catches attack *patterns* the supervised models weren't trained on

A sample is flagged as an alert if **any** of the three agrees it's abnormal.

On the held-out NSL-KDD test set: Random Forest ≈ 77% accuracy, Isolation Forest ≈ 81%. This is the expected range for this benchmark — KDDTest+ deliberately includes attack patterns not present in training, so it measures generalization to unseen attacks rather than a solved classification task.

## How real-time detection works

Your example input was simplified: `IP`, `Packets/sec`, `Failed Logins`, `Protocol`. The full NSL-KDD models expect 41 features (byte counts, service, error rates, etc.), so `app.py` maps your 4 inputs onto a realistic feature vector:

- starts from the **median "normal" traffic** profile learned during training
- overrides `protocol_type`, `num_failed_logins`, and `count`/`srv_count` (driven by packets/sec)
- if packets/sec is very high, also sets DoS-like error-rate/same-service-rate patterns (mirrors a SYN-flood signature)

This is a reasonable demo approximation. **In production**, you'd instead extract the full feature set directly from packet captures/NetFlow using something like CICFlowMeter, rather than approximating it from 4 numbers.

## Setup

```bash
pip install -r requirements.txt
python3 train_model.py      # trains models, writes ./models/
python3 app.py               # starts Flask on http://127.0.0.1:5000
```

Open **http://127.0.0.1:5000** in your browser.

## Using the dashboard

- **Inject Traffic Sample** — manually enter IP / protocol / packets-per-sec / failed logins and classify it
- **Simulate Random Traffic** — generates one random sample (normal, DDoS-like, brute-force-like, or probe-like)
- **Start Auto-Simulation** — fires a new simulated sample every 1.5s so you can watch the live feed fill up
- Stats bar shows total samples analyzed, normal vs. flagged counts, and the top threat type seen

## API reference

```
POST /api/predict
  body: {"ip": "45.33.10.9", "packets_per_sec": 8000, "failed_logins": 50, "protocol": "tcp"}
  returns: {is_attack, attack_type, risk_level, confidence, anomaly_flag, ...}

GET  /api/alerts?limit=50&attacks_only=true
  returns: list of stored alerts, most recent first

GET  /api/stats
  returns: {total_traffic_samples, total_alerts, normal_traffic, by_type}

POST /api/simulate
  generates and classifies one random traffic sample (for demos)
```

## Extending this into the full stack you described

- **Storage**: currently SQLite for portability — swap `sqlite3` calls for `psycopg2`/`mysql-connector` to move to PostgreSQL/MySQL with minimal changes to the query strings.
- **Notifications**: add an email/Slack webhook call inside `predict()` and `simulate()` right after a high/critical-risk alert is stored.
- **Frontend**: the current dashboard is server-rendered HTML/JS; it can be swapped for a React app that polls the same `/api/alerts` and `/api/stats` endpoints.
- **Real traffic ingestion**: replace the simplified `build_feature_vector()` mapping with a real packet-capture pipeline (e.g. `scapy` or `CICFlowMeter`) that computes the actual 41 NSL-KDD-style features per flow.

---

## v2: Retrained on real CICIDS2017 data

`train_model_v2.py` / `app_v2.py` / `templates/dashboard_v2.html` are a second version of this project trained on genuine 2017 network flow captures (via CICFlowMeter) instead of NSL-KDD, covering more realistic, modern attack types.

### Run it
```bash
python3 train_model_v2.py    # trains on data_v2/cicids2017_sample.csv, writes ./models_v2/
python3 app_v2.py            # starts Flask on http://127.0.0.1:5000
```

### What changed vs. v1

**Data.** This release of CICIDS2017 ships as flow-level statistics with IP addresses and the Protocol column already stripped out (common in "ML-ready" releases, done to prevent the model from memorizing specific IPs instead of learning traffic patterns). It has 78 numeric features — packet counts, byte counts, inter-arrival times, TCP flag counts, etc — no protocol_type/service/flag categorical fields like NSL-KDD had, and no login-attempt fields at all.

**Simplified input schema, updated to match.** Since "protocol" and "failed logins" don't exist in this dataset, the dashboard's real-time input is now: Source IP (display label only — this dataset has no IP data, so it's not fed to the model), Destination Port, Flow Duration, Total Packets, and Packets/sec. Packets/sec maps **directly** onto the real `Flow Packets/s` column — this is a genuine improvement over the v1 approximation, where packets/sec had to be heuristically mapped onto NSL-KDD's abstracted `count`/`srv_count` fields.

**Class balance is inverted from real-world traffic, on purpose — know this before quoting the numbers.** The original full dataset is realistically imbalanced (~75% benign, ~25% attacks). To fit Claude's 500MB upload limit, the working copy here (`cicids2017_sample.csv`, 581,632 rows) keeps **every** attack row but downsamples benign traffic to 40,000 — flipping the ratio to roughly 93% attack / 7% benign. `class_weight="balanced"` is used in both Random Forest models and the Decision Tree to compensate during training, but the accuracy figures below should be read with that context, not as "how this would perform on real traffic volumes."

### Accuracy — and why ~99.9% is not as impressive as it looks

| Model | Accuracy |
|---|---|
| Random Forest (binary) | 99.93% |
| Random Forest (attack category) | 99.93% |
| Decision Tree (binary) | 99.78% |
| Isolation Forest (unsupervised) | 49.6% |

The near-perfect supervised accuracy is a **known, documented property of CICIDS2017**, not a sign of an unusually strong model — several of its attack types (PortScan, DDoS, Hulk) have flow signatures so distinct from benign traffic that tree-based models separate them almost trivially. This is one of the dataset's cited limitations in IDS research (compared with NSL-KDD, which is intentionally harder because its test set includes attack types absent from training). Quote this number carefully — say what it means, not just what it is.

**Isolation Forest performing at ~50% (near chance) is expected here, not a bug.** It's trained only on "normal" flows and works by treating rare/different-looking traffic as anomalous — that assumption breaks down when attacks make up 93% of the sample instead of being the rare minority, which is exactly what happened after the benign-downsampling needed to fit the upload limit. It's kept in the ensemble for architectural completeness (and because it's a reasonable talking point about the strengths/limits of unsupervised approaches), but the Random Forest/Decision Tree are doing the real work in this version.

**Heartbleed (11 rows) and Infiltration (36 rows) are too rare to model reliably**, regardless of resampling — this is a real-world data scarcity problem, not something fixable by adjusting the pipeline. Worth mentioning if asked, rather than letting the 99%+ multi-class numbers imply otherwise.

### Update: retrained on the full 2.2M-row dataset for realistic class balance

The models were later retrained on the complete CICIDS2017 sample (2,214,469 rows) instead of the size-constrained 581,632-row downsample described above, restoring the dataset's natural ~75% benign / ~25% attack proportions instead of the inverted 93%/7% ratio.

**This produced one clear improvement and one clear tradeoff:**

| Metric | Downsampled (93% attack) | Full dataset (75% benign) |
|---|---|---|
| Isolation Forest accuracy | 49.6% (near chance) | **79.4%** |
| Botnet precision | 90% | **24%** (recall stayed at 98%) |
| Random Forest / Decision Tree accuracy | ~99.9% | ~99.9% (essentially unchanged) |

**Why Isolation Forest improved:** its core assumption -- normal traffic is the majority, attacks are rare outliers -- only holds when trained on realistic proportions. Restoring the natural ratio fixed this directly.

**Why Botnet precision dropped:** Botnet is an extremely rare class in real proportions (1,966 of 2.2 million rows, ~0.09%). Against a much larger, more diverse pool of normal traffic, the model has a harder time drawing a clean boundary around such a rare class, leading to more false "Botnet" labels on traffic that isn't actually Botnet -- even though it still catches 98% of genuine Botnet traffic.

**This is the version currently deployed.** It was chosen over the downsampled version because it reflects the dataset's real, intended proportions rather than an artifact of a file-size constraint, and because a non-functional Isolation Forest (one third of the ensemble) was judged a more significant flaw than reduced precision on one rare category. This is a classic, well-documented tension in imbalanced classification -- fixing overall class balance for one purpose can measurably affect a specific rare-class metric elsewhere -- and is worth stating plainly rather than only reporting whichever number looks best.

### Files added for v2
| File | Purpose |
|---|---|
| `data_v2/cicids2017_sample.csv` | Real CICIDS2017 flow data (class-imbalance-adjusted sample) |
| `train_model_v2.py` | Preprocessing + training for the CICIDS2017 schema |
| `app_v2.py` | Flask API matching the new (port/duration/packets-based) input schema |
| `templates/dashboard_v2.html` | Dashboard updated for the new fields and attack categories |
| `models_v2/` | Saved models trained on CICIDS2017, plus `demo_samples.csv` |
| `ids_alerts_v2.db` | Separate SQLite DB for v2 alerts (created on first run) |

### "Simulate" vs. manual "Analyze Sample" -- an important distinction

These two buttons work differently, and it's worth understanding why:

- **"Analyze Sample" (manual form / `/api/predict`)** takes only 4 numbers (port, duration, packet count, packets/sec) and approximates the other ~74 features from a "typical normal" baseline. This is realistic for what a simplified dashboard input can actually provide, but it means only very extreme patterns (huge packet rate, specific ports) reliably get flagged -- subtler attack types like PortScan or slow DoS variants won't show up correctly here, because their real signatures differ from normal traffic across many features at once, not just the 4 exposed in the form.

- **"Simulate Random Traffic" / "Start Auto-Simulation" (`/api/simulate`)** classifies real, held-out CICIDS2017 flow records (saved to `demo_samples.csv` during training, one class of rows the model never saw while training) -- genuine 78-feature vectors, not approximations. This is what correctly shows the full range of categories (DoS, DDoS, PortScan, Botnet, Infiltration, Heartbleed) and reflects the model's actual measured accuracy.

If asked in an interview why these two buttons can classify the same-looking numbers differently: that's the honest answer -- one is a live approximation from limited real-time signals, the other replays genuine unseen data. Real IDS deployments face exactly this tension between what you can cheaply measure in real time versus the full feature set a model was trained on.

### Forcing a specific category (demo aid)

Two categories -- Heartbleed (3 of 139 demo rows) and Infiltration (11 of 139) -- are naturally rare, mirroring their scarcity in the real dataset (11 and 36 rows respectively, out of 2.2 million). Random "Simulate" clicks only have a ~2-8% chance of hitting them.

The dashboard's **"Force Specific Category"** dropdown (`/api/simulate_category`) lets you pick any of the 7 categories and reliably classify a real example of it on demand -- useful for demoing the full range of detections without relying on random luck.

---

## Real packet capture (`capture_traffic.py`)

This captures **your own machine's live network traffic**, reconstructs it into flows, computes real CICFlowMeter-style statistics, and classifies each flow through the running Flask app -- no simulation, no approximation.

### Legal note
Only run this on a network/device you own or are explicitly authorized to monitor. Capturing traffic on a network you don't control (shared Wi-Fi, a workplace network without permission, etc.) is illegal in most places. This tool is intended for monitoring your own machine's own traffic only.

### Setup
```bash
pip install -r requirements-capture.txt
```
**Windows only:** also install [Npcap](https://npcap.com/#download) (check "Install Npcap in WinPcap API-compatible mode" during setup) -- this is the packet-capture driver Scapy needs on Windows.

### Run it
1. Start the main app first: `python app_v2.py`
2. In a **separate terminal, run as Administrator** (right-click Command Prompt/PowerShell -> "Run as administrator"):
   ```bash
   python capture_traffic.py --duration 30
   ```
3. While it's capturing, browse the web, stream a video, etc, to generate real traffic
4. After the capture window ends, it reconstructs each flow and sends it to `/api/predict_full` -- results print in the terminal AND appear live on the dashboard

### What it computes vs. approximates
Directly computed from real packets: packet/byte counts, inter-arrival times, TCP flag counts, header lengths, init window sizes, and more -- roughly 60 of the model's 78 features come from genuine traffic. A handful of fields CICFlowMeter derives from long-duration bulk-transfer and idle-period detection (active/idle timing, bulk transfer rate) are set to 0/baseline, since reliably detecting those needs longer, more complex flow-state tracking than this simplified capture tool implements.

### What you'll likely see
Your own normal browsing will (correctly) classify as Normal Traffic most of the time. To see an attack-like flow flagged safely and legally, you can run a port scan **against your own machine** (e.g. `nmap -sS 127.0.0.1` or `nmap -sS <your own local IP>`, if you have nmap installed) while the capture tool is running -- that generates a real PortScan-shaped flow pattern without touching anyone else's system.

### A real finding worth knowing: concept drift on live traffic

Early testing of this tool against real 2026 browser traffic showed some normal HTTPS flows getting flagged as "Unknown Suspicious Activity." Investigating this surfaced a genuinely interesting, defensible finding rather than a bug to hide:

CICIDS2017 was captured in **2017** -- different TLS versions, different browser/OS network stack behavior than 2026 traffic has today. This is a well-documented phenomenon in ML security research called **concept drift**: a model's learned notion of "normal" is tied to the specific era/environment it was trained on, and real-world traffic characteristics shift over time. A 9-year-old dataset won't perfectly represent today's traffic patterns.

The original ensemble logic flagged an alert if **any one of the three models** objected. Since Isolation Forest was already known to be the weakest model here (~50% accuracy, due to the class-imbalance from downsampling), it was the most likely source of these false positives on live traffic it wasn't well-calibrated for.

**Fix:** the ensemble now requires **at least 2 of the 3 models to agree** before raising an alert, rather than acting on any single model's opinion. This meaningfully reduces false positives from Isolation Forest's known weakness while still catching genuine attacks (verified against both the manual-input DDoS test and the real-data `/api/simulate` samples, which still correctly flag ~85-90% of attack rows).

This is worth mentioning directly in an interview: *"I tested my model against my own live traffic and found false positives caused by concept drift between 2017 training data and 2026 real traffic -- then fixed it with a majority-vote ensemble instead of hiding or ignoring the issue."* That's a stronger, more credible story than a demo that only shows curated success cases.

---

## Deployment

The Flask app (`app_v2.py`) can be deployed as a public web service using [Render](https://render.com) (free tier):

1. Push this repo to GitHub
2. On Render: **New +** -> **Web Service** -> connect this repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app_v2:app`
5. Deploy -- Render gives you a public URL (e.g. `https://netsentry-ids.onrender.com`)

**Important limitation:** only the **Simulate** and **manual "Analyze Sample"** features work on a public deployment. The **real packet capture tool (`capture_traffic.py`) cannot run on a hosted server** -- it reads packets directly from your own machine's network adapter, which a cloud server has no access to. That tool is designed to be run locally, on your own PC, as a live demo during an interview rather than as part of the hosted link.

Free-tier hosting spins down after ~15 minutes of inactivity, so the first request after idle time may take 30-60 seconds to respond -- this is expected free-tier behavior, not a bug.


