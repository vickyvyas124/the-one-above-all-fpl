"""Adapter for StatsBomb's research-oriented open-data GitHub repository.

This adapter does not imply current-season Premier League coverage. Its league
and season catalogue is source-controlled and must be inspected before use.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Mapping

import requests

from ..contract import (
    CONTRACT_VERSION,
    EvidenceQuality,
    PlayerMatchObservation,
    validate_observation,
)


SOURCE_ID = "statsbomb_open_data"
DEFAULT_BASE_URL = "https://raw.githubusercontent.com/hudl/open-data/master/data"
ADAPTER_VERSION = "1.0.0"
MAX_RESPONSE_BYTES = 30 * 1024 * 1024
DEFENSIVE_EVENT_TYPES = {"Interception", "Block", "Clearance", "Ball Recovery", "Pressure"}
ON_TARGET_OUTCOMES = {"Goal", "Saved"}
NOT_COMPLETED_STATUSES = {
    "scheduled", "not started", "in progress", "postponed", "abandoned", "cancelled",
}


class SourceDataError(RuntimeError):
    """The source response could not be safely parsed or validated."""


@dataclass(frozen=True)
class RawDocument:
    url: str
    body: bytes
    payload: Any
    retrieved_at_utc: str
    source_timestamp: str | None
    sha256: str


@dataclass(frozen=True)
class MatchBundle:
    source_id: str
    competition_id: int
    season_id: int
    match_id: int
    match: Mapping[str, Any] | None
    events: tuple[Mapping[str, Any], ...] | None
    lineups: tuple[Mapping[str, Any], ...] | None
    documents: Mapping[str, RawDocument]
    source_timestamp: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _raw_document(url: str, body: bytes, payload: Any, *, source_timestamp: str | None) -> RawDocument:
    return RawDocument(
        url=url, body=bytes(body), payload=payload, retrieved_at_utc=_utc_now(),
        source_timestamp=source_timestamp, sha256=hashlib.sha256(body).hexdigest(),
    )


def _list_payload(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise SourceDataError(f"StatsBomb {label} response must be a list of objects.")
    return value


def _seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        minute, second = str(value).split(":", 1)
        result = int(minute) * 60 + float(second)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _event_player_id(event: Mapping[str, Any]) -> str | None:
    player = event.get("player")
    if not isinstance(player, dict) or player.get("id") is None:
        return None
    return str(player["id"])


class StatsBombOpenDataAdapter:
    """Fetch a selected completed match with bounded, explicit requests.

    StatsBomb asks publishers of research/analysis based on Open Data to credit
    StatsBomb and use its logo. Raw data should remain outside this public repo.
    """

    def __init__(self, *, session: Any | None = None, base_url: str = DEFAULT_BASE_URL,
                 timeout_seconds: float = 20.0,
                 player_id_map: Mapping[str, int] | None = None):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.player_id_map = {str(key): int(value) for key, value in (player_id_map or {}).items()}

    def _fetch(self, path: str, *, expected: type, source_timestamp: str | None = None) -> RawDocument | None:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self.session.get(
                url, timeout=self.timeout_seconds,
                headers={"User-Agent": f"The-One-Above-All-FPL-Football-Intelligence/{ADAPTER_VERSION}"},
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            body = bytes(response.content)
            if len(body) > MAX_RESPONSE_BYTES:
                raise SourceDataError(f"StatsBomb response exceeded {MAX_RESPONSE_BYTES} bytes: {path}")
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise SourceDataError(f"StatsBomb response was not valid UTF-8 JSON: {path}") from exc
            if not isinstance(payload, expected):
                raise SourceDataError(f"StatsBomb response had the wrong top-level shape: {path}")
            return _raw_document(url, body, payload, source_timestamp=source_timestamp)
        except SourceDataError:
            raise
        except requests.RequestException as exc:
            raise SourceDataError(f"StatsBomb request failed for {path}: {exc}") from exc

    def fetch_competitions(self) -> tuple[list[dict[str, Any]], RawDocument]:
        document = self._fetch("competitions.json", expected=list)
        if document is None:
            raise SourceDataError("StatsBomb competition catalogue is unavailable.")
        return _list_payload(document.payload, "competition catalogue"), document

    def fetch_match_bundle(self, competition_id: int, season_id: int, match_id: int) -> MatchBundle:
        if min(int(competition_id), int(season_id), int(match_id)) <= 0:
            raise ValueError("Competition, season, and match identifiers must be positive.")
        match_path = f"matches/{int(competition_id)}/{int(season_id)}.json"
        match_document = self._fetch(match_path, expected=list)
        if match_document is None:
            raise SourceDataError("StatsBomb season match list is unavailable.")
        match_rows = _list_payload(match_document.payload, "season match list")
        matches = [row for row in match_rows if str(row.get("match_id")) == str(match_id)]
        if len(matches) > 1:
            raise SourceDataError(f"Match {match_id} is duplicated in the requested competition and season.")
        match = matches[0] if matches else None
        if match is None:
            raise SourceDataError(f"Match {match_id} is not listed in the requested competition and season.")
        source_timestamp = (match.get("last_updated") or match.get("match_updated")
                            or match.get("match_available"))
        status = str(match.get("match_status") or "").strip().lower()
        if status not in {"available", "completed"} | NOT_COMPLETED_STATUSES:
            raise SourceDataError(f"StatsBomb match {match_id} has an unknown or missing completion status.")
        if status not in {"available", "completed"}:
            return MatchBundle(SOURCE_ID, int(competition_id), int(season_id), int(match_id), match,
                               None, None, {"matches": match_document}, source_timestamp)

        documents = {"matches": match_document}
        event_document = self._fetch(f"events/{int(match_id)}.json", expected=list,
                                     source_timestamp=source_timestamp)
        lineup_document = self._fetch(f"lineups/{int(match_id)}.json", expected=list,
                                      source_timestamp=source_timestamp)
        if event_document:
            documents["events"] = event_document
        if lineup_document:
            documents["lineups"] = lineup_document
        events = _list_payload(event_document.payload, "events") if event_document else None
        lineups = _list_payload(lineup_document.payload, "lineups") if lineup_document else None
        return MatchBundle(SOURCE_ID, int(competition_id), int(season_id), int(match_id), match,
                           tuple(events) if events is not None else None,
                           tuple(lineups) if lineups is not None else None,
                           documents, source_timestamp)

    def normalize_match(self, bundle: MatchBundle) -> tuple[PlayerMatchObservation, ...]:
        match = bundle.match
        if not isinstance(match, Mapping) or bundle.events is None or bundle.lineups is None:
            raise SourceDataError("A completed match with event and lineup data is required for normalization.")
        home, away = match.get("home_team"), match.get("away_team")
        if not isinstance(home, dict) or not isinstance(away, dict):
            raise SourceDataError("StatsBomb match is missing home/away team identities.")
        team_by_id = {}
        for team in (home, away):
            if team.get("home_team_id") is not None:
                team_by_id[str(team["home_team_id"])] = (str(team.get("home_team_name") or ""),
                                                           str(away.get("away_team_name") or ""))
            if team.get("away_team_id") is not None:
                team_by_id[str(team["away_team_id"])] = (str(team.get("away_team_name") or ""),
                                                           str(home.get("home_team_name") or ""))
        if not team_by_id or any(not club.strip() or not opponent.strip() for club, opponent in team_by_id.values()):
            raise SourceDataError("StatsBomb match team identities are incomplete.")
        date = str(match.get("match_date") or "")
        if len(date) != 10:
            raise SourceDataError("StatsBomb match date is missing or malformed.")
        fixture_id = str(match.get("match_id") or bundle.match_id)
        season = str((match.get("season") or {}).get("season_name") or f"statsbomb:{bundle.season_id}")
        competition_round = str(match.get("match_week")) if match.get("match_week") is not None else None
        retrieved_at = max(document.retrieved_at_utc for document in bundle.documents.values())
        raw_hash = hashlib.sha256("".join(
            bundle.documents[name].sha256 for name in sorted(bundle.documents)
        ).encode("ascii")).hexdigest()

        events_by_player: dict[str, list[Mapping[str, Any]]] = {}
        max_match_clock = 0.0
        seen_event_ids: set[str] = set()
        for event in bundle.events:
            if event.get("id") is None:
                raise SourceDataError("StatsBomb event is missing its stable event ID.")
            event_id = str(event["id"])
            if event_id in seen_event_ids:
                raise SourceDataError(f"StatsBomb event ID is duplicated: {event_id}.")
            seen_event_ids.add(event_id)
            player_id = _event_player_id(event)
            if player_id is not None:
                events_by_player.setdefault(player_id, []).append(event)
            minute, second = event.get("minute"), event.get("second", 0)
            try:
                clock = float(minute) * 60.0 + float(second)
                if math.isfinite(clock):
                    max_match_clock = max(max_match_clock, clock)
            except (TypeError, ValueError):
                continue

        lineup_players = []
        seen_player_ids: set[str] = set()
        for team in bundle.lineups:
            team_id = team.get("team_id")
            if team_id is None or str(team_id) not in team_by_id or not isinstance(team.get("lineup"), list):
                raise SourceDataError("StatsBomb lineup response contains an invalid team or player list.")
            for player in team["lineup"]:
                if not isinstance(player, dict) or player.get("player_id") is None or not player.get("player_name"):
                    raise SourceDataError("StatsBomb lineup contains a player without a stable source identity.")
                player_id = str(player["player_id"])
                if player_id in seen_player_ids:
                    raise SourceDataError(f"StatsBomb lineup player ID is duplicated: {player_id}.")
                seen_player_ids.add(player_id)
                lineup_players.append((str(team_id), player))
        if not lineup_players:
            raise SourceDataError("StatsBomb lineup response contains no player records.")

        shot_xg_by_id: dict[str, float] = {}
        for event in bundle.events:
            shot = event.get("shot")
            if isinstance(shot, dict) and event.get("id") is not None:
                try:
                    value = float(shot["statsbomb_xg"])
                    if math.isfinite(value) and value >= 0:
                        shot_xg_by_id[str(event["id"])] = value
                except (KeyError, TypeError, ValueError):
                    pass

        output = []
        for team_id, player in lineup_players:
            player_id = str(player["player_id"])
            name = str(player["player_name"])
            club, opponent = team_by_id[team_id]
            segments = []
            position_rows = player.get("positions")
            if isinstance(position_rows, list):
                for segment in position_rows:
                    if not isinstance(segment, dict):
                        continue
                    segments.append({
                        "position": segment.get("position"),
                        "position_id": segment.get("position_id"),
                        "from": segment.get("from"), "from_period": segment.get("from_period"),
                        "to": segment.get("to"), "to_period": segment.get("to_period"),
                        "start_reason": segment.get("start_reason"),
                        "end_reason": segment.get("end_reason"),
                    })
            started = any(str(row.get("start_reason") or "").lower() == "starting xi" for row in segments)
            substituted_on = any(str(row.get("start_reason") or "").lower().startswith("substitution - on")
                                 for row in segments)
            minutes = None
            if isinstance(position_rows, list):
                total_seconds = 0.0
                intervals_known = True
                for segment in position_rows:
                    if not isinstance(segment, dict):
                        continue
                    start = _seconds(segment.get("from"))
                    end = _seconds(segment.get("to"))
                    if end is None and str(segment.get("end_reason") or "").lower() == "final whistle":
                        end = max_match_clock if max_match_clock > 0 else None
                    if start is None or end is None:
                        intervals_known = False
                        break
                    total_seconds += max(0.0, end - start)
                if not position_rows:
                    minutes = 0.0
                elif intervals_known:
                    minutes = round(total_seconds / 60.0, 2)

            player_events = events_by_player.get(player_id, [])
            shots = [event for event in player_events if isinstance(event.get("shot"), dict)]
            if shots and all(event.get("id") is not None and str(event["id"]) in shot_xg_by_id for event in shots):
                xg = round(sum(shot_xg_by_id[str(event["id"])] for event in shots), 6)
            elif not shots:
                xg = 0.0
            else:
                xg = None
            on_target_known = all(isinstance(event.get("shot"), dict)
                                  and isinstance(event["shot"].get("outcome"), dict)
                                  for event in shots)
            shots_on_target = (sum(str(event["shot"]["outcome"].get("name")) in ON_TARGET_OUTCOMES
                                   for event in shots) if on_target_known else None)
            passes = [event.get("pass") for event in player_events if isinstance(event.get("pass"), dict)]
            shot_assists = [row for row in passes if row.get("shot_assist") is True]
            goal_assists = [row for row in passes if row.get("goal_assist") is True]
            chances_created = sum(bool(row.get("shot_assist") or row.get("goal_assist")) for row in passes)
            xa_values = [shot_xg_by_id.get(str(row.get("assisted_shot_id")))
                         for row in shot_assists if row.get("assisted_shot_id") is not None]
            xa = (0.0 if not shot_assists else
                  round(sum(xa_values), 6) if len(xa_values) == len(shot_assists)
                  and all(value is not None for value in xa_values) else None)
            key_passes = len(shot_assists)
            attack_events = 0
            box_events = 0
            for event in player_events:
                location = event.get("location")
                if not isinstance(location, list) or len(location) < 2:
                    continue
                try:
                    x, y = float(location[0]), float(location[1])
                except (TypeError, ValueError):
                    continue
                # This is an event-location proxy, not a complete tracking-data touch count.
                attack_events += int(x >= 80.0)
                box_events += int(x >= 102.0 and 18.0 <= y <= 62.0)
            defensive_actions = 0
            for event in player_events:
                event_type = str((event.get("type") or {}).get("name") or "")
                if event_type in DEFENSIVE_EVENT_TYPES:
                    defensive_actions += 1
                elif event_type == "Duel" and str((event.get("duel") or {}).get("type", {}).get("name") or "") == "Tackle":
                    defensive_actions += 1

            position = segments[0].get("position") if segments else None
            missing = []
            started_value = started if isinstance(position_rows, list) else None
            substitute_value = substituted_on if isinstance(position_rows, list) else None
            for field_name, value in (("position", position), ("minutes", minutes), ("started", started_value),
                                      ("substitute", substitute_value), ("shots_on_target", shots_on_target),
                                      ("xg", xg), ("xa", xa)):
                if value is None:
                    missing.append(field_name)
            # Progressive events require a validated attack-direction normalization;
            # this adapter leaves them unknown rather than guessing from raw x values.
            missing.extend(("progressive_carries", "progressive_passes"))
            quality = EvidenceQuality.INCOMPLETE if missing else EvidenceQuality.HIGH
            observation = PlayerMatchObservation(
                schema_version=CONTRACT_VERSION, source_id=SOURCE_ID, source_player_id=player_id,
                fpl_player_id=self.player_id_map.get(player_id), season=season,
                gameweek=None, competition_round=competition_round, fixture_id=fixture_id,
                match_date=date, player_name=name, club=club, opponent=opponent,
                position=str(position) if position is not None else None, minutes=minutes,
                started=started_value,
                substitute=substitute_value,
                shots=len(shots), shots_on_target=shots_on_target, xg=xg, xa=xa,
                key_passes=key_passes, chances_created=chances_created,
                touches_attacking_area=attack_events, touches_in_box=box_events,
                progressive_carries=None, progressive_passes=None,
                defensive_actions=defensive_actions, role_segments=tuple(segments),
                source_timestamp=bundle.source_timestamp, retrieved_at_utc=retrieved_at,
                source_url=bundle.documents["matches"].url, raw_bundle_sha256=raw_hash,
                completeness=quality, missing_fields=tuple(sorted(set(missing))),
                metric_definitions={
                    "xg": "Sum of StatsBomb shot.statsbomb_xg for this source player in this match.",
                    "xa": "Proxy: sum of linked StatsBomb shot xG for pass events marked shot_assist; null if a link is missing.",
                    "key_passes": "Count of source pass events with shot_assist=true; source definition, not cross-vendor normalized.",
                    "chances_created": "Count of unique source pass events marked shot_assist or goal_assist.",
                    "touches_attacking_area": "Count of this player's event records at x>=80 on the 120x80 source pitch; event-location proxy, not full touch tracking.",
                    "touches_in_box": "Count of this player's event records at x>=102 and 18<=y<=62; event-location proxy, not full touch tracking.",
                    "defensive_actions": "Count of source Interception, Block, Clearance, Ball Recovery, Pressure and Tackle-duel events.",
                    "progressive_carries": "Unknown until source direction normalization is validated.",
                    "progressive_passes": "Unknown until source direction normalization is validated.",
                },
            )
            validate_observation(observation)
            output.append(observation)
        return tuple(output)
