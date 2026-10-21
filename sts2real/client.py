"""STS2MCP REST client — minimal typed wrapper over the game bridge API."""

from __future__ import annotations

import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:15526"


class Sts2McpError(RuntimeError):
    pass


class Sts2McpClient:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 10.0):
        self.base_url = base_url
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = self.base_url + path
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise Sts2McpError(f"HTTP {e.code}: {body[:500]}") from e
        except urllib.error.URLError as e:
            raise Sts2McpError(f"connection failed: {e}") from e

    def health(self) -> dict:
        return self._request("GET", "/")

    def state(self) -> dict:
        return self._request("GET", "/api/v1/singleplayer?format=json")

    def action(self, payload: dict) -> dict:
        return self._request("POST", "/api/v1/singleplayer", payload)

    def menu_select(self, option: str, **extra) -> dict:
        return self.action({"action": "menu_select", "option": option, **extra})

    # --- convenience flows ---

    def start_seeded_custom_run(self, character: str, seed: str) -> dict:
        """Navigate: main -> singleplayer -> custom -> character select with seed."""
        s = self.state()
        assert s.get("state_type") == "menu", f"expected menu, got {s.get('state_type')}"
        self.menu_select("singleplayer")
        self.menu_select("custom")
        return self.menu_select(character.lower(), seed=seed)

    def combat_fingerprint(self) -> dict:
        """Extract the deterministic fingerprint of the current combat state."""
        s = self.state()
        p = s.get("player", {})
        enemies = s.get("enemies", [])
        return {
            "seed": s.get("run", {}).get("seed"),
            "floor": s.get("run", {}).get("floor"),
            "hp": p.get("hp"),
            "max_hp": p.get("max_hp"),
            "hand": [c.get("id") for c in p.get("hand", [])],
            "energy": p.get("energy"),
            "enemies": [
                {"id": e.get("id"), "hp": e.get("hp"), "intent": e.get("intent")}
                for e in enemies
            ],
        }
