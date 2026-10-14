from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "Data"
PROCESSED_DIR = ROOT / "Data" / "processed"
RAW_GLOB = "*_LoL_esports_match_data_from_OraclesElixir.csv"

# Multi-league events: never used as a team's home league.
# Chosen because >60% of their games pair teams from different home leagues
# (see the data report), plus domestic cups that mix main and academy teams.
EVENT_LEAGUES = {
    "WLDs", "MSI", "FST", "EWC",          # major internationals
    "EM", "EUM",                          # EMEA Masters (ERL teams)
    "WSCI", "ASCI", "ASI", "AC", "AmCh",  # smaller cross-region events
    "LTA",                                # 2025 LTA cross-conference finals
    "KeSPA Cup", "DCup",                  # domestic cups
}

# The events that matter for Worlds backtesting / pick'ems.
MAJOR_INTL = {"WLDs", "MSI", "FST", "EWC"}

# Leagues with a direct MSI/Worlds slot (home leagues of MSI/Worlds teams 2022-2025,
# after LEAGUE_ALIAS). TCL left out: it went to Worlds in 2022 but has played as
# an EMEA regional league since 2023. Everything else starts below these in Elo.
MAJOR_LEAGUES = {"LCK", "LPL", "LEC", "LCS", "CBLOL", "LLA", "LCP", "PCS", "VCS", "LJL", "LCO"}

# Leagues that were renamed: LCS/CBLOL ran as "LTA North/South" in 2025,
# Spain's LVP SL became LES in 2026.
LEAGUE_ALIAS = {"LTA N": "LCS", "LTA S": "CBLOL", "LES": "LVP SL"}

# Columns that only have values on player rows.
PLAYER_ONLY_COLS = {
    "participantid", "position", "playername", "playerid", "champion", "url",
}
