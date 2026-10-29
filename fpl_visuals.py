"""Safe official Premier League image URL helpers for squad visuals."""

import re


_PLAYER_PHOTO_ID = re.compile(r"^(\d+)\.(?:jpg|png)$", re.IGNORECASE)


def player_headshot_url(player):
    """Build the official headshot URL from the FPL ``photo`` identifier.

    Current bootstrap data uses values such as ``226597.jpg``. The numeric
    stem is the Premier League photo identifier; it is not the season's FPL
    element ID.
    """
    if not isinstance(player, dict):
        return None
    match = _PLAYER_PHOTO_ID.fullmatch(str(player.get("photo", "")).strip())
    if not match or int(match.group(1)) <= 0:
        return None
    return ("https://resources.premierleague.com/premierleague/photos/players/"
            f"110x140/p{match.group(1)}.png")


def club_badge_url(teams_by_id, team_id):
    """Map an FPL team ID to its official Premier League badge URL."""
    if not isinstance(teams_by_id, dict):
        return None
    try:
        team_id = int(team_id)
    except (TypeError, ValueError, OverflowError):
        return None
    team = teams_by_id.get(team_id)
    if team is None:
        team = teams_by_id.get(str(team_id))
    if not isinstance(team, dict):
        return None
    try:
        code = int(team.get("code"))
    except (TypeError, ValueError, OverflowError):
        return None
    if code <= 0:
        return None
    return f"https://resources.premierleague.com/premierleague/badges/50/t{code}.png"


def player_initials(player):
    """Return two readable initials for the headshot fallback."""
    if not isinstance(player, dict):
        return "?"
    parts = [str(player.get(key, "")).strip() for key in ("first_name", "second_name")]
    initials = "".join(part[0] for part in parts if part)
    if initials:
        return initials[:2].upper()
    web_name = str(player.get("web_name", "")).strip()
    return web_name[:2].upper() if web_name else "?"
