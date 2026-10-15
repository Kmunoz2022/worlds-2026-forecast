# Design decisions

Why this project is built the way it is: what I chose, what I tried and rejected, and the bugs I found along the way. All numbers are out-of-sample unless stated.

## Highlights

- **I optimized for calibrated probabilities, not accuracy.** Pre-game, about 66% is the ceiling at Worlds, and the model reaches it. The 95%-accuracy goal is met by the live model, not by pushing the pre-game model. → [Goal and evaluation](#goal-and-evaluation)
- **Cross-region ratings.** Teams rarely play other regions, so a plain Elo is *worse than a coin flip* at Worlds (52.8%). Splitting each rating into a team part and a league part fixed that. → [Ratings](#ratings-elo-with-cross-region-strength)
- **The bug where a relegation series moved a whole region.** A bottom LCS team's promotion matches lifted every LCS team by ~200 rating points. → [Bugs I found](#bugs-i-found-and-fixed)
- **Backtesting the Crystal Ball forecasts showed they were weak, so I published them as long shots.** The champion that dominates Worlds is usually created by the Worlds patch itself. → [Crystal Ball](#crystal-ball-forecasts)
- **Live in-game predictions that run in the visitor's browser**, straight from the LoL Esports broadcast feed: calls above 90% are right 95–97% of the time. → [Live predictions](#live-in-game-predictions)
- **A public prediction ledger.** Every 2026 prediction is committed before the match and graded after, so the record can't be fitted to results. → [Ledger](#the-public-prediction-ledger)

---

## Goal and evaluation

**Calibration over raw accuracy.** The original goal was "95% accuracy". Before a game starts that isn't possible: at Worlds, teams are close in strength. So I judge the pre-game model on **log loss** and **Brier score** (do 70% predictions win about 70% of the time?) and report accuracy alongside.

**The pre-game ceiling, measured.** If a model's probabilities are right, its expected accuracy equals the average favourite's probability. For Worlds 2022–25 that's 66.1%, and the model scored 66.0%. About a third of Worlds games have a favourite under 60%, so accuracy can only go up with new information. The live model adds that information, which is where the 95% goal is actually met.

**No leakage.**
- Every feature uses only games played before the one being predicted.
- The models are retrained every month and only ever predict the next month (walk-forward).
- Worlds games were never used for tuning or for choosing a model; they're the final test.

**Baselines.** Coin flip, blue-side rate, a plain Elo, and a "stronger region" rule (pick the team whose region has the better international record). The region rule turned out to be the toughest baseline at Worlds.

## Data

- **Source:** Oracle's Elixir match data, 2022–2026: 53.5k pro games across ~85 leagues. Team-level rows only; player rows are used later for player questions.
- **Cleaning:**
  - Dropped 3 games with no recorded winner.
  - Filled 1.9k missing team IDs from the team name.
  - Used the game **date** instead of the `year` column, which is a season label (December games are marked as the next year).
- **Series aren't in the data**, so I rebuilt them from the same two teams playing consecutive game numbers in the same league: 30k series, with 1.8% too ambiguous to classify (Bo2 draws, missing games).
- **"International" means the two teams' home leagues differ, not the league label.** Oracle's Elixir files regional qualifiers under "Worlds" (e.g. LPL vs LPL). Each team's home league is its most-played domestic league that year.
- **Storage:** Parquet instead of CSV, so types are kept and loads are ~10× faster. The full rebuild takes ~20 seconds.

## Ratings: Elo with cross-region strength

**Team strength = team rating + league offset.**
- Every game updates both teams' ratings, and cross-league games move them more.
- League offsets move only on international games, so one MSI upset nudges every team in both regions.
- Ratings pull back toward average each new year to reflect roster turnover.
- New teams adjust faster for their first 10 games.

**Why:** without linking, ratings from different regions aren't comparable. A plain Elo scored 52.8% at Worlds, below a coin flip. With linking it scored 62.8%.

**Guardrails added after finding problems** (details in the bugs table):
- Leagues without an MSI/Worlds slot start below the majors.
- A major region's strength only changes from games against other major regions. I tested letting the tuner choose how much minor-league games count: it pushed the weight *up* and made Worlds predictions worse, so this is a fixed design rule rather than a tuned number.

**Tuning target:** log loss on MSI/EWC/First Stand games, domestic games in major leagues, and major-vs-minor games. Never Worlds.

## Game model and calibration

**Features (~24 per game):**
- the Elo parts (team gap and region gap, separately)
- each region's international record
- recent form: exponentially weighted gold spent, towers, dragons, barons, kills, and wins against expectation
- experience, rest days, and the series score so far

**Left out on purpose: gold/XP at 15 minutes.** About 85% of LPL games lack those stats. The model would learn "stats missing" instead of "team quality" and quietly treat LPL teams differently.

**Model choice: logistic regression, international games weighted 5×.**
- Gradient boosting was as good or worse everywhere, so I kept the simpler model: it's naturally calibrated and every feature has one readable coefficient.
- I added a rule to keep it unless a challenger wins on the validation set by more than 0.005 log loss, so noise can't flip the model choice between data refreshes.

**Calibration:** temperature scaling. One multiplier on the log-odds, fit on earlier out-of-sample international predictions. Raw predictions were about 10% overconfident on international games. Isotonic regression was rejected because it overfits a few hundred games.

| Worlds 2022–25 (430 games) | Accuracy | Log loss |
|---|---|---|
| **Final model** | **66.0%** | **0.608** |
| Stronger-region rule | 65.6% | 0.614 |
| Elo only | 62.8% | 0.622 |
| Coin flip | 54.9% | 0.693 |

On all 4,226 cross-region games since 2023 the gap is clearer: 69.5% vs 64.8% for the region rule.

## Series and tournament simulation

**Games in a series aren't independent.** Bo3s end 2-0 65% of the time vs 55% if each game were a coin flip, and Bo5 sweeps happen 39% vs 30%. A team that's better on the day stays better. I model this with a **beta-binomial**: each series draws its own edge around the predicted probability. The spread (κ = 6) is fitted on 11k series.
- It keeps single-game odds unchanged.
- It reproduces real score distributions. For favourites at 60–70% per game, it predicts 31% 3-0 against 33% actual.
- 3-0 is often the single most likely score but only ~25–30% likely, so the site shows the chance of every score rather than one "predicted score".

**Tournament simulator:** 20,000 runs of the real format:
- Play-In: double elimination, all Bo5
- Swiss: Bo1, with Bo3 for advancement and elimination matches, same-record pairing, no same-region round 1
- Knockouts: Bo5

Where official draw rules weren't published I used stated approximations. I checked past Worlds for team-level variance beyond what the model already expects and found none, so I didn't add extra noise. That check used only 84 team-tournaments, so heavy favourites (Gen.G ~42% to win the title) may still be slightly overrated.

**Live series odds** use the same model, updated with each finished game (a 60% favourite up 2-0 is 95%; down 0-2, 13%).

## Crystal Ball forecasts

These are 17 prediction questions on the LoL Esports site, about champions, players, games and teams.

**Champions.** I backtested champion-meta forecasts on Worlds 2022–2025 before trusting them. The champion that dominates Worlds is usually a Worlds-patch power pick (Aatrox went from 30% presence before Worlds to 98% during it; Orianna from 5% to 94%). Even after tuning (games on the newest patch count 10×, plus wide uncertainty), the favourite was right in only **1 of 12** past questions. **I published these as long shots, with the backtest shown next to them**, rather than presenting false confidence.

**Players and games:** layered on the tournament simulation. For each simulated game, each team draws one of its own recent games with the same result, so kills, assists and team kills stay consistent. Three corrections came from comparing against past Worlds:
- **Weak-league inflation.** A support from a weaker region ranked near the top for KDA by stomping domestic opponents. Player stats now come only from games against Worlds-level opponents.
- **Rare events need Worlds-level history.** 15-kill games happen in 0.2% of Worlds games but 2–3% at MSI/EWC. Base rates now use Worlds plus MSI at half weight, which moved "15+ kills: Yes" from 85% to a coin flip.
- **Game length depends on the mismatch.** Heavy favourites' wins average 28.6 minutes vs ~32.5 for close games. The shortest-game forecast now models that instead of relying on a 15-game sample.

Every answer comes with a plain-language "why" generated from the simulation's own numbers. The submitted picks are frozen in `ledger/` and tracked as the event runs.

## Live in-game predictions

**Model:** one logistic regression for any minute of the game. Inputs: pre-game log-odds plus gold, kill and CS leads, each also scaled by game time. Trained on snapshots at 0/10/15/20/25 minutes. XP isn't used because the live feed doesn't carry it. Tested on 18.7k games from 2025–26:

| Minute | Accuracy | Games with a ≥90% call | Those calls right |
|---|---|---|---|
| 10 | 71.6% | 10% | 96.1% |
| 15 | 75.4% | 24% | 94.8% |
| 20 | 79.7% | 38% | 95.4% |
| 25 | 83.8% | 48% | 96.6% |

One model with time-scaled leads, rather than separate per-minute models, gives smooth predictions at any minute.

**Feed:** the unofficial endpoints that lolesports.com uses. Quirks I had to handle:
- There's no game clock, so I find the start by stepping back and narrowing to the first frame.
- Asked for a time after a game ended, the feed silently returns the game's *final* frames.
- The feed can have mid-game gaps (12 minutes in one game).
- Some second-tier events have stats disabled.

Teams are identified by their players' names rather than team names (the feed shows "TLAW Morgan", not "Team Liquid").

**In the browser, no server.** The feed allows requests from any website, so the public site runs the whole live pipeline in the visitor's browser: it finds live Worlds games, the game start, the teams, the live win chance and the series odds, and refreshes every 10 seconds. Opened mid-game, it fills in the earlier minutes so the chart shows the whole game. The model is 7 coefficients exported from Python, and the same formulas are used on both sides. A replay switch (`?match=<id>&at=<time>`) runs any finished match as if it were live, which is how I test it.

**Validation:** replaying a finished match (LCS final, Team Liquid 3–1 LYON) through the same code: all four games followed, and the live favourite at the end of each game was the actual winner.

**Timeline archive.** Oracle's Elixir only has snapshots at 10/15/20/25 minutes, with no timed objectives. The feed has the whole game, so a script archives every finished Worlds game at one sample per game-minute: each team's gold, kills, towers, dragons, barons and inhibitors, and each player's gold, CS, level, K/D/A, items, wards, damage share and kill participation (about 12 KB per game). The feed keeps games for at least a year, so past events can be backfilled. This is groundwork for a live model that also uses objectives, and for measuring how teams play from behind (comebacks, lost leads).

**Draft experiment.** Adding champion strength in the current meta and each player's record on their champion improved pre-game predictions on all 2025–26 games (accuracy 64.9% → 65.8%, log loss 0.626 → 0.617, 18.8k games), but not on international games (1,967 games, slightly worse). Worlds plays on its own patch, players' champion records come from domestic play, and top teams draft carefully against each other. So the draft isn't in the model yet. It will be retested on Worlds-patch games once enough are played.

## The public prediction ledger

Grading only means something if the predictions were provably made first.
- `ledger/predictions.jsonl` is **append-only**: each entry is timestamped before the match, and entries are never edited.
- After the match, each played series is graded against the most recent prediction made before it started.
- For the double-elimination Play-In, I log **every possible pairing in advance**, so later bracket matches have a prediction on record even when two matches are played the same day.
- Submitted pick'em and Crystal Ball picks are frozen in `ledger/*_locked.json`. Later model runs update the live odds but never the picks.
- The repository's commit history adds public timestamps.

## Engineering

- **One-command refresh** (`scripts/refresh.py`). After downloading new data it rebuilds data, ratings, model and simulation, grades the ledger, logs newly possible matches, updates the trackers and exports the site. It refuses to run on empty input files, which happened once during setup.
- **Website:** one static page plus one JSON file, plain JavaScript with hand-drawn SVG charts, no framework. Six small pages don't need React, and every number on the site comes from the exported JSON, none typed by hand. It has light and dark themes and works at phone width.
- **Why static instead of Streamlit:** free hosting, nothing to keep running, and a site rather than a dashboard. Live predictions still work because the browser itself reads the feed.
- **Hosting:** GitHub Pages, deployed by a GitHub Actions workflow on every push to `main`.
- **Design:** the interface follows Apple's design guidance:
  - system type, with tracking that tightens as the size grows
  - a translucent header
  - a segmented control whose thumb moves on an interruptible spring
  - swiping between tabs on touch screens, with 1:1 tracking, rubber-banding at the ends, and momentum projection to decide if a flick changes tab
  - respect for the reduce-motion, reduce-transparency and increase-contrast settings

  I tested the motion with simulated pointer input. That caught two bugs: a spring could get a negative time step, and a swipe could start half-initialized.
- **Secrets:** the live-feed API key (a public web-client key) is kept out of the code and read from an environment variable or a git-ignored local file.
- **Stack:** Python 3.13, pandas, NumPy, scikit-learn, Parquet.

## Bugs I found and fixed

| Bug | How I found it | Fix |
|---|---|---|
| A French second-tier team ranked #1 in the world | Sanity-checking the top of the Elo table | Minor leagues start below the majors; they barely play major teams, so their level floated freely |
| A relegation series lifted the whole LCS by ~200 points | Tracing a sudden rating jump game by game | Major regions are only rated against other major regions |
| Regional qualifiers counted as Worlds games | "Worlds 2026" showed 12 LPL-vs-LPL games | Worlds starts at its first cross-region game; the same filter is applied everywhere the history is used |
| The Worlds test set dropped real same-region Worlds matches | Reviewing test-set sizes | Test on every Worlds game after the start date |
| Patch "15.20" read as "15.2" (January's patch) | A champion backtest that sorted patches oddly | Read patches as text, not numbers |
| Model choice flipped on noise (0.003 log loss on 321 games) | A data refresh changed the selected model | A switching margin |
| Weak-region players topped the player-stat forecasts | Checking leaders against past Worlds | Use only games against Worlds-level opponents |
| The shortest-game forecast rested on 15 games | Explaining the pick in plain language | Model game length from the mismatch and team pace |
| The live replay started at the wrong time | Minute −0.3 with a 6,500 gold lead | The feed returns final frames for late timestamps; reject frames far from the requested time |
| Joining a game after a feed outage reset its clock | The live page showed 4:18 with a 17–5 kill score | A 12-minute gap in the feed looked like "before the game"; start detection now rides through up to 18 minutes of silence |

## What I'd do next

- Grade the 2026 ledger and the live calls after Worlds, and publish the results.
- Add tests (series maths, ledger grading, feed parsing, no-leakage checks) and CI.
- Model rating uncertainty (Glicko-2) for teams with few games.
- Add roster-change features from player data.
