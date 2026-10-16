"""Download team logos and player photos for the website (see docs/DECISIONS.md).

Logos and photos come from the LoL Esports API (the images lolesports.com
shows), shrunk to small PNGs in site/img/ so the site doesn't depend on
another server for them. Champion and item icons are not downloaded: the site
loads them from Riot's Data Dragon CDN, whose version is saved here.
Writes config/team_assets.json. Re-run if a roster or logo changes.

Riot allows fan projects to use these assets (see README credits).
Uses macOS `sips` to resize.

Usage: .venv/bin/python scripts/fetch_assets.py
"""
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lolpred import livefeed  # noqa: E402

IMG = ROOT / "site" / "img"
OUT = ROOT / "config" / "team_assets.json"
DDRAGON = "https://ddragon.leagueoflegends.com"
# Our team name -> lolesports team slug (the main roster, not academies).
SLUGS = {"Gen.G": "geng", "Hanwha Life Esports": "hanwha-life-esports", "T1": "t1", "Dplus KIA": "dwg-kia",
         "Anyone's Legend": "anyones-legend", "Bilibili Gaming": "bilibili-gaming", "Top Esports": "top-esports",
         "Invictus Gaming": "invictus-gaming", "G2 Esports": "g2-esports", "Movistar KOI": "mad-lions",
         "Karmine Corp": "karmine-corp", "Team Liquid": "team-liquid", "LYON": "lyon-gaming", "Cloud9": "cloud9",
         "Team Secret Whales": "team-secret-whales", "CTBC Flying Oyster": "ctbc-flying-oyster",
         "MVK Esports": "saigon-buffalo-esports", "LØS": "los", "FURIA": "furia"}
# White logos that vanish on the light theme: these teams' alternate logo is a dark
# version; the others are drawn as a dark silhouette in light mode instead.
LIGHT_ALT = {"Hanwha Life Esports", "Dplus KIA", "G2 Esports", "MVK Esports"}
LIGHT_INK = {"Team Liquid", "Team Secret Whales"}
ROLES = {"top": "Top", "jungle": "Jungle", "mid": "Mid", "bottom": "Bot", "support": "Support"}


def download(url, dest, size):
    """Fetch an image and shrink it to fit size x size (keeps PNG transparency)."""
    url = url.replace("http://", "https://")
    with tempfile.NamedTemporaryFile(suffix=Path(url).suffix or ".png") as tmp:
        req = urllib.request.Request(url, headers={"User-Agent": "lolpred-worlds-forecast"})
        with urllib.request.urlopen(req, timeout=30) as r:
            tmp.write(r.read())
        tmp.flush()
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["sips", "-s", "format", "png", "-Z", str(size), tmp.name, "--out", str(dest)],
                       check=True, capture_output=True)


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def main():
    teams = livefeed._get(f"{livefeed.API}/getTeams?hl=en-US")["data"]["teams"]
    by_slug = {t["slug"]: t for t in teams}
    out = {"teams": {}, "players": {}}
    for name, slug in SLUGS.items():
        t = by_slug[slug]
        logo = IMG / "teams" / f"{slug}.png"
        download(t["image"], logo, 96)
        out["teams"][name] = {"code": t["code"], "slug": slug, "esports_id": t["id"],
                              "logo": f"img/teams/{slug}.png"}
        if name in LIGHT_ALT:
            download(t["alternativeImage"], IMG / "teams" / f"{slug}-light.png", 96)
            out["teams"][name]["logo_light"] = f"img/teams/{slug}-light.png"
        if name in LIGHT_INK:
            out["teams"][name]["logo_ink"] = True
        for p in t.get("players", []):
            if p.get("role") not in ROLES or not p.get("image"):
                continue
            photo = IMG / "players" / f"{p['id']}.png"
            try:
                download(p["image"], photo, 128)
            except Exception as e:  # a missing photo isn't worth failing over
                print(f"  no photo for {p['summonerName']}: {e}")
                continue
            out["players"][p["summonerName"]] = {"team": name, "role": ROLES[p["role"]],
                                                 "photo": f"img/players/{p['id']}.png"}
        print(f"{name}: logo + {sum(v['team'] == name for v in out['players'].values())} players")

    version = get_json(f"{DDRAGON}/api/versions.json")[0]
    champs = get_json(f"{DDRAGON}/cdn/{version}/data/en_US/champion.json")["data"]
    out["ddragon"] = {"version": version, "base": f"{DDRAGON}/cdn/{version}/img",
                      "champions": {c["name"]: c["id"] for c in champs.values()}}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"Data Dragon {version}: {len(champs)} champions. Wrote {OUT}")


if __name__ == "__main__":
    main()
