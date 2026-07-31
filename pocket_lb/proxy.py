from __future__ import annotations

import itertools
import json
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from pocket_lb.config import CloudflareAccount, Settings, load_settings


RETRY_STATUSES = {400, 401, 403, 404, 408, 409, 410, 425, 429, 500, 502, 503, 504}
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
}


class AccountPool:
    def __init__(self, accounts: list[CloudflareAccount]) -> None:
        self._accounts = accounts
        self._cursor = itertools.count()
        self._rate_limited_until: dict[str, float] = {}

    def report_rate_limit(self, account: CloudflareAccount, wait_seconds: float = 60.0) -> None:
        self._rate_limited_until[account.account_id] = time.time() + wait_seconds

    def next_attempts(self, max_attempts: int, usage_tracker: 'UsageTracker | None' = None) -> list[CloudflareAccount]:
        candidates = self._accounts
        now = time.time()
        
        # Filter out currently rate-limited accounts
        available_candidates = [
            acc for acc in candidates
            if self._rate_limited_until.get(acc.account_id, 0.0) < now
        ]
        
        # Fallback to all candidates if all are rate-limited to at least try something
        if not available_candidates:
            available_candidates = candidates
            
        candidates = available_candidates

        if usage_tracker:
            valid = []
            for acc in self._accounts:
                rem = usage_tracker.snapshot(acc).get("remaining_tokens")
                if rem is None or rem > 0:
                    valid.append(acc)
            if valid:
                candidates = valid

        if not candidates:
            return []

        attempts = len(candidates)
        start = next(self._cursor)
        return [candidates[(start + offset) % len(candidates)] for offset in range(attempts)]


class UsageTracker:
    def __init__(self, accounts: list[CloudflareAccount], state_path: Path | None = None) -> None:
        self.state_path = state_path
        self._usage: dict[str, dict[str, object]] = {}
        now = time.time()
        
        # Load persisted state if exists
        if self.state_path and self.state_path.exists():
            try:
                import json
                self._usage = json.loads(self.state_path.read_text())
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to load state: {e}")

        # Initialize any missing accounts
        for account in accounts:
            if account.account_id not in self._usage:
                self._usage[account.account_id] = {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "unknown_token_responses": 0,
                    "period_started_at": now,
                    "last_used_at": None,
                }

    def _save_state(self) -> None:
        if self.state_path:
            try:
                import json
                self.state_path.write_text(json.dumps(self._usage))
            except Exception:
                pass

    def record(self, account: CloudflareAccount, payload: object | None) -> None:
        item = self._usage.setdefault(
            account.account_id,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "unknown_token_responses": 0,
                "period_started_at": time.time(),
                "last_used_at": None,
            },
        )
        self._reset_if_needed(account, item)
        item["requests"] += 1
        item["last_used_at"] = time.time()

        usage = payload.get("usage") if isinstance(payload, dict) else None
        if not isinstance(usage, dict):
            item["unknown_token_responses"] += 1
            self._save_state()
            return

        item["prompt_tokens"] += _safe_int(usage.get("prompt_tokens"))
        item["completion_tokens"] += _safe_int(usage.get("completion_tokens"))
        item["total_tokens"] += _safe_int(usage.get("total_tokens"))
        self._save_state()

    def snapshot(self, account: CloudflareAccount) -> dict[str, object]:
        item = self._usage.setdefault(
            account.account_id,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "unknown_token_responses": 0,
                "period_started_at": time.time(),
                "last_used_at": None,
            },
        )
        self._reset_if_needed(account, item)
        total_tokens = int(item["total_tokens"])
        remaining = None if account.token_limit is None else max(0, account.token_limit - total_tokens)
        reset_at = None
        if account.reset_period_hours:
            reset_at = float(item["period_started_at"]) + (account.reset_period_hours * 3600)
        return {
            **item,
            "remaining_tokens": remaining,
            "reset_at": reset_at,
        }

    def _reset_if_needed(self, account: CloudflareAccount, item: dict[str, object]) -> None:
        if not account.reset_period_hours:
            return
        reset_at = float(item["period_started_at"]) + (account.reset_period_hours * 3600)
        if time.time() < reset_at:
            return
        item.update(
            {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "unknown_token_responses": 0,
                "period_started_at": time.time(),
                "last_used_at": None,
            }
        )


class RequestLog:
    def __init__(self, limit: int = 80, log_path: Path | None = None) -> None:
        self.log_path = log_path
        self._entries: deque[dict[str, object]] = deque(maxlen=limit)
        
        if self.log_path and self.log_path.exists():
            try:
                import json
                saved = json.loads(self.log_path.read_text())
                if isinstance(saved, list):
                    for entry in reversed(saved):  # We snapshot as list (latest first), so insert reversed
                        self._entries.appendleft(entry)
            except Exception:
                pass

    def _save(self) -> None:
        if self.log_path:
            try:
                import json
                self.log_path.write_text(json.dumps(self.snapshot()))
            except Exception:
                pass

    def record(self, entry: dict[str, object]) -> None:
        self._entries.appendleft({"timestamp": time.time(), **entry})
        self._save()

    def snapshot(self) -> list[dict[str, object]]:
        return list(self._entries)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    pool = AccountPool(settings.accounts)
    
    state_path = settings.config_path.with_name("state.json") if settings.config_path else None
    usage_tracker = UsageTracker(settings.accounts, state_path=state_path)
    
    log_path = settings.config_path.with_name("request_log.json") if settings.config_path else None
    request_log = RequestLog(log_path=log_path)
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal client
        client = httpx.AsyncClient(timeout=timeout)
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(title="Pocket-LB", version="0.1.0", lifespan=lifespan)

    @app.get("/logo.png")
    async def get_logo():
        from fastapi.responses import FileResponse
        import os
        path = "hud-dashboard/public/logo.png"
        if os.path.exists(path):
            return FileResponse(path)
        from fastapi import Response
        return Response(status_code=404)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(saved: bool = False, error: str | None = None) -> str:
        return _dashboard_html(settings, usage_tracker, request_log, saved=saved, error=error)

    @app.get("/setup")
    async def setup_form() -> RedirectResponse:
        return RedirectResponse(url="/?tab=settings")

    @app.post("/setup")
    async def save_setup(request: Request) -> RedirectResponse:
        body = (await request.body()).decode()
        form = parse_qs(body, keep_blank_values=True)
        accounts = []

        account_count = max(1, _safe_int(_form_value(form, "account_count"), 1))
        for index in range(1, account_count + 1):
            account_id = _form_value(form, f"account_id_{index}")
            api_token = _form_value(form, f"api_token_{index}")
            name = _form_value(form, f"name_{index}") or f"user-{index}@example.com"
            token_limit = _form_value(form, f"token_limit_{index}")
            reset_period_hours = _form_value(form, f"reset_period_hours_{index}")
            if account_id or api_token:
                if not account_id or not api_token:
                    from urllib.parse import quote
                    err_msg = quote(f"Account #{index} needs both account ID and API token.")
                    return RedirectResponse(url=f"/?error={err_msg}", status_code=303)
                account = {"name": name, "account_id": account_id, "api_token": api_token}
                if token_limit:
                    account["token_limit"] = int(token_limit)
                if reset_period_hours:
                    account["reset_period_hours"] = int(reset_period_hours)
                accounts.append(account)

        if not accounts:
            from urllib.parse import quote
            return RedirectResponse(url=f"/?error={quote('Add at least one Cloudflare account.')}", status_code=303)

        config = {
            "host": settings.host,
            "port": settings.port,
            "request_timeout_seconds": settings.request_timeout_seconds,
            "max_attempts": min(settings.max_attempts, len(accounts)) or 1,
            "model_mapping": settings.model_mapping,
            "accounts": accounts,
        }
        settings.config_path.write_text(json.dumps(config, indent=2) + "\n")
        
        settings.accounts = [
            CloudflareAccount(
                name=acc["name"],
                account_id=acc["account_id"],
                api_token=acc["api_token"],
                token_limit=acc.get("token_limit"),
                reset_period_hours=acc.get("reset_period_hours"),
            )
            for acc in accounts
        ]
        settings.max_attempts = config["max_attempts"]
        pool._accounts = settings.accounts
        
        return RedirectResponse(url="/?saved=1", status_code=303)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "ok": True,
            "accounts": [account.name for account in settings.accounts],
            "max_attempts": settings.max_attempts,
        }

    @app.get("/usage")
    async def usage() -> dict[str, object]:
        accounts = []
        for account in settings.accounts:
            item = usage_tracker.snapshot(account)
            accounts.append(
                {
                    "name": account.name,
                    "account_id": _mask_account_id(account.account_id),
                    "token_limit": account.token_limit,
                    "reset_period_hours": account.reset_period_hours,
                    **item,
                }
            )
        return {"accounts": accounts, "request_log": request_log.snapshot()}


    @app.get("/v1/models")
    async def get_models():
        if client is None:
            return JSONResponse({"error": "Proxy client is not ready."}, status_code=503)
        if not settings.accounts:
            return JSONResponse({"error": "No Cloudflare accounts configured.", "setup_url": "/setup"}, status_code=428)
            
        account = settings.accounts[0]
        url = f"https://api.cloudflare.com/client/v4/accounts/{account.account_id}/ai/models/search"
        headers = {"Authorization": f"Bearer {account.api_token}"}
        
        try:
            resp = await client.get(url, headers=headers, timeout=10.0)
            resp.raise_for_status()
            cf_models = [m["name"] for m in resp.json().get("result", []) if "name" in m]
        except Exception as e:
            print(f"[LB] Error fetching models from Cloudflare: {e}", flush=True)
            cf_models = []
            
        all_model_names = list(settings.model_mapping.keys()) + cf_models
        
        seen = set()
        unique_models = []
        for m in all_model_names:
            if m not in seen:
                seen.add(m)
                unique_models.append(m)
                
        models_data = [
            {
                "id": m_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "cloudflare" if m_name.startswith("@cf/") else "pocket_lb"
            }
            for m_name in unique_models
        ]
        
        return JSONResponse({
            "object": "list",
            "data": models_data
        })

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy(path: str, request: Request) -> Response:
        if client is None:
            return JSONResponse({"error": "Proxy client is not ready."}, status_code=503)
        if not settings.accounts:
            return JSONResponse(
                {"error": "No Cloudflare accounts configured.", "setup_url": "/setup"},
                status_code=428,
            )

        body = await request.body()
        started_at = time.time()
        requested_model = None
        mapped_model = None
        if request.method == "POST" and "application/json" in request.headers.get("content-type", ""):
            try:
                payload = json.loads(body)
                requested_model = payload.get("model") if isinstance(payload, dict) else None
                if "model" in payload and payload["model"] in settings.model_mapping:
                    old_model = payload["model"]
                    payload["model"] = settings.model_mapping[payload["model"]]
                    mapped_model = payload["model"]
                    print(f"[LB] Rewriting model {old_model} -> {payload['model']}", flush=True)
                    body = json.dumps(payload).encode("utf-8")
                    # Update content-length if it exists
                    if "content-length" in request.headers:
                        incoming_headers = dict(_forward_headers(request))
                        incoming_headers["content-length"] = str(len(body))
                        _request_headers_cache = incoming_headers
            except Exception as e:
                print(f"[LB] Error parsing JSON/mapping model: {e}", flush=True)
                pass
                
        incoming_headers = locals().get("_request_headers_cache", _forward_headers(request))
        last_response: httpx.Response | None = None
        last_error: Exception | None = None
        attempts_log = []

        attempted_accounts = pool.next_attempts(settings.max_attempts, usage_tracker)
        for i, account in enumerate(attempted_accounts):
            headers = dict(incoming_headers)
            headers["authorization"] = f"Bearer {account.api_token}"
            headers["cf-aig-authorization"] = f"Bearer {account.api_token}"
            
            print(f"[LB] Attempt {i+1}: Trying account {account.name}...", flush=True)

            try:
                upstream_response = await client.send(
                    client.build_request(
                        request.method,
                        account.endpoint(path),
                        params=request.query_params,
                        content=body,
                        headers=headers,
                    ),
                    stream=True,
                )
            except httpx.HTTPError as exc:
                print(f"[LB] Attempt {i+1}: Account {account.name} failed with network error: {exc}", flush=True)
                attempts_log.append({"account": account.name, "status": "network-error"})
                last_error = exc
                continue

            print(f"[LB] Attempt {i+1}: Account {account.name} returned status {upstream_response.status_code}", flush=True)
            attempts_log.append({"account": account.name, "status": upstream_response.status_code})

            if upstream_response.status_code not in RETRY_STATUSES:
                request_log.record(
                    {
                        "method": request.method,
                        "path": f"/v1/{path}",
                        "model": requested_model,
                        "mapped_model": mapped_model,
                        "status": upstream_response.status_code,
                        "account": account.name,
                        "attempts": attempts_log,
                        "duration_ms": int((time.time() - started_at) * 1000),
                    }
                )
                return await _tracked_response(upstream_response, account, usage_tracker)

            if upstream_response.status_code == 429:
                # Tell the pool to avoid this account for a short time
                # If there is a Retry-After header we can use it, otherwise default to 60s
                retry_after = upstream_response.headers.get("retry-after")
                wait_seconds = 60.0
                if retry_after:
                    try:
                        wait_seconds = float(retry_after)
                    except ValueError:
                        pass
                pool.report_rate_limit(account, wait_seconds)

            last_response = upstream_response
            await upstream_response.aclose()

        if last_response is not None:
            tried_names = ", ".join(a.name for a in attempted_accounts)
            request_log.record(
                {
                    "method": request.method,
                    "path": f"/v1/{path}",
                    "model": requested_model,
                    "mapped_model": mapped_model,
                    "status": last_response.status_code,
                    "account": None,
                    "attempts": attempts_log,
                    "duration_ms": int((time.time() - started_at) * 1000),
                }
            )
            return JSONResponse(
                {"error": f"All Cloudflare accounts failed or were rate limited. Tried: {tried_names}", "last_status": last_response.status_code},
                status_code=last_response.status_code,
            )

        request_log.record(
            {
                "method": request.method,
                "path": f"/v1/{path}",
                "model": requested_model,
                "mapped_model": mapped_model,
                "status": 502,
                "account": None,
                "attempts": attempts_log,
                "duration_ms": int((time.time() - started_at) * 1000),
                "error": str(last_error) if last_error else None,
            }
        )
        return JSONResponse(
            {"error": "All Cloudflare accounts failed.", "detail": str(last_error) if last_error else None},
            status_code=502,
        )

    return app


def _forward_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "authorization"
    }


def _response_headers(response: httpx.Response, account_name: str) -> dict[str, str]:
    headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }
    headers["x-pocket-lb-account"] = account_name
    return headers


async def _tracked_response(response: httpx.Response, account: CloudflareAccount, usage_tracker: UsageTracker) -> Response:
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        return _stream_response(response, account, usage_tracker)

    body = await response.aread()
    payload = None
    if "json" in content_type:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
    usage_tracker.record(account, payload)
    await response.aclose()
    return Response(
        content=body,
        status_code=response.status_code,
        headers=_response_headers(response, account.name),
        media_type=response.headers.get("content-type"),
    )


def _stream_response(response: httpx.Response, account: CloudflareAccount, usage_tracker: UsageTracker) -> StreamingResponse:
    async def body() -> AsyncIterator[bytes]:
        buffer = b""
        try:
            async for chunk in response.aiter_bytes():
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.startswith(b"data: "):
                        data_str = line[6:].strip()
                        if data_str and data_str != b"[DONE]":
                            try:
                                payload = json.loads(data_str.decode("utf-8"))
                                if payload.get("usage"):
                                    usage_tracker.record(account, payload)
                            except Exception:
                                pass
                yield chunk
        finally:
            await response.aclose()

    return StreamingResponse(
        body(),
        status_code=response.status_code,
        headers=_response_headers(response, account.name),
        media_type=response.headers.get("content-type"),
    )


def _dashboard_html(settings: Settings, usage_tracker: UsageTracker, request_log: RequestLog, saved: bool = False, error: str | None = None) -> str:
    base_url = f"http://{settings.host}:{settings.port}/v1"
    status_label = "Online" if settings.accounts else "Setup required"
    status_class = "status-ok" if settings.accounts else "status-warn"
    snapshots = [(account, usage_tracker.snapshot(account)) for account in settings.accounts]
    
    existing = settings.accounts or [CloudflareAccount(name="", account_id="", api_token="")]
    account_rows = "".join(_account_setup_block(index, account) for index, account in enumerate(existing, start=1))
    notice = ""
    if saved:
        notice = "<div class=\"notice success\">Saved successfully. Your endpoint is now using the new accounts.</div>"
    elif error:
        notice = f"<div class=\"notice error\">{escape(error)}</div>"
    total_observed = sum(int(item["total_tokens"]) for _, item in snapshots)
    total_limit = sum(account.token_limit or 0 for account, _ in snapshots)
    total_remaining = None if total_limit == 0 else max(0, total_limit - total_observed)
    total_requests = sum(int(item["requests"]) for _, item in snapshots)
    total_unknown = sum(int(item["unknown_token_responses"]) for _, item in snapshots)
    configured_quota_count = sum(1 for account, _ in snapshots if account.token_limit)
    total_usage_percent = 0 if total_limit == 0 else min(100, int((total_observed / total_limit) * 100))
    rows = "".join(
        f"""
        <div class="account-card">
          <div class="account-top">
            <div>
              <b>{escape(account.name)}</b>
              <code>{escape(_mask_account_id(account.account_id))}</code>
            </div>
            <span class="account-status">Active</span>
          </div>
          <div class="account-total">
            <strong>{_format_int(int(item['total_tokens']))}</strong>
            <span>observed tokens</span>
          </div>
          <div class="meter"><i style="width: {_usage_percent(account, item)}%"></i></div>
          <div class="quota-grid">
            <span>Prompt <b>{_format_int(int(item['prompt_tokens']))}</b></span>
            <span>Completion <b>{_format_int(int(item['completion_tokens']))}</b></span>
            <span>Limit <b>{_format_optional_int(account.token_limit)}</b></span>
            <span>Remaining <b>{_format_optional_int(item['remaining_tokens'])}</b></span>
            <span>Requests <b>{_format_int(int(item['requests']))}</b></span>
            <span>Reset <b>{_format_reset(item['reset_at'])}</b></span>
          </div>
          <div class="account-foot">
            <span>Unknown-token responses: <b>{_format_int(int(item['unknown_token_responses']))}</b></span>
            <span>Last used: <b>{_format_time_ago(item['last_used_at'])}</b></span>
          </div>
        </div>
        """
        for account, item in snapshots
    ) or """
        <div class="empty-state">
          <b>No accounts configured</b>
          <p>Add Cloudflare account IDs and API tokens locally. The proxy will stay locked until setup is complete.</p>
          <a class="button" href="/setup">Open Setup</a>
        </div>
    """

    return _codex_shell(
        title="pocketLB",
        subtitle="API Load Balancer",
        shell_class="dashboard-shell",
        body=f"""
        <header class="dashboard-hero">
          <div>
            <h2>Dashboard</h2>
            <p class="muted">Real-time metrics and account distribution.</p>
          </div>
          <div class="hero-actions">
            <span class="status {status_class}">{status_label}</span>
          </div>
        </header>

        <div id="tab-dashboard" class="tab-panel" style="display: block;">
          <section class="stats">
            <div class="card stat-card">
              <div class="stat-header">
                <span class="stat-title">Active Accounts</span>
                <div class="stat-icon pink-icon">
                  <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle></svg>
                </div>
              </div>
              <div class="stat-value"><b id="stat-accounts">{len(settings.accounts)}</b></div>
            </div>
            <div class="card stat-card">
              <div class="stat-header">
                <span class="stat-title">Requests Proxied</span>
                <div class="stat-icon cyan-icon">
                  <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline></svg>
                </div>
              </div>
              <div class="stat-value"><b id="stat-requests">{_format_int(total_requests)}</b></div>
            </div>
            <div class="card stat-card">
              <div class="stat-header">
                <span class="stat-title">Total Tokens Used</span>
                <div class="stat-icon pink-icon">
                  <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
                </div>
              </div>
              <div class="stat-value"><b id="stat-tokens">{_format_int(total_observed)}</b></div>
            </div>
            <div class="card stat-card">
              <div class="stat-header">
                <span class="stat-title">Total Estimated Remaining</span>
                <div class="stat-icon cyan-icon">
                  <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none"><rect x="2" y="2" width="20" height="20" rx="5" ry="5"></rect><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path></svg>
                </div>
              </div>
              <div class="stat-value"><b id="stat-remaining">{_format_optional_int(total_remaining)}</b></div>
            </div>
          </section>

          <section class="dashboard-section" style="margin-top: 24px; padding: 0; background: transparent; border: 0; box-shadow: none;">
            <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 24px;">
              
              <!-- Total System Quota -->
              <div class="card" style="display: flex; flex-direction: column;">
                <div class="section-title">
                  <h3>System Quota Utilization</h3>
                </div>
                <div style="flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 24px 0;">
                  <div style="position: relative; width: 220px; height: 110px; display: flex; justify-content: center; align-items: flex-end; overflow: hidden; margin-bottom: 24px;">
                    <svg viewBox="0 0 100 50" width="220" height="110" style="position: absolute; bottom: 0;">
                      <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="var(--line)" stroke-width="12" stroke-linecap="round" />
                      <path id="quota-donut" d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="var(--accent)" stroke-width="12" stroke-linecap="round"
                            stroke-dasharray="{int(126 * total_usage_percent / 100)} 126" />
                    </svg>
                    <div style="text-align: center; margin-bottom: 12px; z-index: 2;">
                      <div id="quota-percent" style="font-size: 36px; font-weight: 800; color: var(--text); font-family: var(--mono); line-height: 1;">{total_usage_percent}%</div>
                      <div style="font-size: 13px; color: var(--muted); margin-top: 4px;">Utilized</div>
                    </div>
                  </div>
                  
                  <div style="width: 100%; display: flex; flex-direction: column; gap: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 13px;">
                      <span style="color: var(--muted);">Total Limit</span>
                      <b style="color: var(--text); font-family: var(--mono);">{_format_optional_int(total_limit) if total_limit else 'Unlimited'}</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 13px;">
                      <span style="color: var(--muted);">Tokens Used</span>
                      <b id="quota-observed" style="color: var(--text); font-family: var(--mono);">{_format_int(total_observed)}</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 13px;">
                      <span style="color: var(--muted);">Uncounted (Stream/Unknown)</span>
                      <b id="quota-unknown" style="color: var(--text); font-family: var(--mono);">{_format_int(total_unknown)}</b>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Account Distribution List -->
              <div class="card" style="display: flex; flex-direction: column;">
                <div class="section-title">
                  <h3>Account Distribution</h3>
                  <span id="quota-configured" style="font-size: 13px;">{configured_quota_count}/{len(snapshots)} configured</span>
                </div>
                <div class="bar-chart" id="account-bars" style="flex: 1; display: flex; flex-direction: column; gap: 16px; margin-top: 16px; overflow-y: auto; max-height: 280px; padding-right: 8px;">
                  {_account_bars(snapshots)}
                </div>
              </div>
            </div>
          </section>

          <section class="dashboard-section request-log-section">
            <div class="section-title">
              <h3>Request Log</h3>
              <span id="request-log-count">{len(request_log.snapshot())} recent</span>
            </div>
            <div class="request-log" id="request-log">
              {_request_log_rows(request_log.snapshot())}
            </div>
          </section>
        </div>

        <div id="tab-accounts" class="tab-panel" style="display: none;">
          <section class="dashboard-section accounts-section">
            <div class="section-title">
              <h3>Accounts</h3>
              <span>Round-robin failover pool</span>
            </div>
            <div class="accounts" id="accounts-list">{rows}</div>
          </section>
        </div>

        <div id="tab-settings" class="tab-panel" style="display: none;">
          <section class="dashboard-section endpoint-panel endpoint-bottom">
            <div class="section-title">
              <h3>OpenAI-compatible endpoint</h3>
              <span>/v1</span>
            </div>
            <label>Base URL
              <div class="copyline"><code>{escape(base_url)}</code><button type="button" data-copy="{escape(base_url)}">Copy</button></div>
            </label>
            <div class="endpoint-meta">
              <span>Host <b>{escape(settings.host)}</b></span>
              <span>Port <b>{settings.port}</b></span>
              <span>Retries <b>{settings.max_attempts}</b></span>
              <span>Timeout <b>{settings.request_timeout_seconds:g}s</b></span>
            </div>
          </section>

          <section class="dashboard-section model-mapping-panel">
            <div class="section-title">
              <h3>Model Mappings</h3>
              <span>Configured client aliases</span>
            </div>
            <div style="font-size: 13px; line-height: 1.6; margin-bottom: 12px; color: var(--text-muted);">
              When an AI client requests one of these models, the proxy rewrites it to the Cloudflare model shown. Edit <code>config.json</code> to modify these aliases.
            </div>
            <div style="display: grid; gap: 8px; font-family: var(--font-mono); font-size: 12px;">
              {"".join(f'<div style="display: flex; gap: 12px; align-items: center; padding: 8px 12px; background: var(--bg-rail); border-radius: 6px;"><strong style="color: var(--text); min-width: 150px;">{escape(client_model)}</strong> <span style="color: var(--text-muted);">➔</span> <span>{escape(cf_model)}</span></div>' for client_model, cf_model in settings.model_mapping.items())}
            </div>
          </section>

          <section class="dashboard-section">
            <div class="section-title">
              <h3>Setup Accounts</h3>
              <span>Local config.json</span>
            </div>
            {notice}
            <form method="post" action="/setup" class="stack" id="setup-form">
              <input type="hidden" name="account_count" id="account-count" value="{len(existing)}">
              <div id="account-setup-list">{account_rows}</div>
              <div class="actions" style="margin-top: 10px;">
                <button class="button secondary" type="button" id="add-account">Add Another Account</button>
                <button class="button" type="submit">Save Configuration</button>
              </div>
            </form>
          </section>

          <section class="dashboard-section info-strip" style="flex-direction: row; flex-wrap: wrap;">
            <p style="flex: 1; min-width: 300px;">Token counts are local observations from provider <code>usage</code> fields. Streaming responses or providers that omit usage are tracked as unknown-token responses. <span id="usage-sync">Waiting for backend sync...</span></p>
            <div class="actions">
              <a class="button secondary" href="/usage">Usage JSON</a>
              <a class="button secondary" href="/health">Health JSON</a>
            </div>
          </section>
        </div>

        {_dashboard_script()}
        """,
    )


def _request_log_rows(entries: list[dict[str, object]]) -> str:
    if not entries:
        return '<div class="chart-empty">No proxied requests yet. Send traffic to <code>/v1</code> to populate this log.</div>'

    rows = []
    for entry in entries:
        status = entry.get("status")
        model = escape(str(entry.get("model") or "Unknown model"))
        mapped_model = entry.get("mapped_model")
        if mapped_model:
            model = f"{model} &rarr; {escape(str(mapped_model))}"
        attempts = entry.get("attempts")
        attempt_text = "No attempts recorded"
        if isinstance(attempts, list):
            attempt_text = " · ".join(
                f"{escape(str(attempt.get('account', 'unknown')))}:{escape(str(attempt.get('status', 'unknown')))}"
                for attempt in attempts
                if isinstance(attempt, dict)
            ) or attempt_text
        rows.append(
            f"""
            <div class="request-log-row">
              <div class="request-log-main">
                <span class="request-status {_request_status_class(status)}">{escape(str(status or 'n/a'))}</span>
                <div>
                  <b>{escape(str(entry.get('method') or 'GET'))} {escape(str(entry.get('path') or '/v1'))}</b>
                  <span>{model}</span>
                </div>
              </div>
              <div class="request-log-meta">
                <span>{escape(str(entry.get('account') or 'No account accepted'))}</span>
                <span>{_format_int(int(entry.get('duration_ms') or 0))}ms</span>
                <span>{_format_time_ago(entry.get('timestamp'))}</span>
              </div>
              <code>{attempt_text}</code>
            </div>
            """
        )
    return "".join(rows)


def _request_status_class(status: object) -> str:
    code = _safe_int(status)
    if 200 <= code < 300:
        return "ok"
    if code == 429:
        return "warn"
    if code >= 400:
        return "error"
    return "muted"


def _dashboard_script() -> str:
    return """<script>
      const $ = (id) => document.getElementById(id);
      const themeButton = $('theme-toggle');
      const applyTheme = (theme) => {
        document.documentElement.dataset.theme = theme;
        localStorage.setItem('pocket-lb-theme', theme);
        if (themeButton) {
          themeButton.textContent = theme === 'dark' ? 'Light theme' : 'Dark theme';
          themeButton.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`);
        }
      };
      applyTheme(localStorage.getItem('pocket-lb-theme') || 'light');
      const formatInt = (value) => Number(value || 0).toLocaleString();
      const formatOptionalInt = (value) => value === null || value === undefined ? 'Not set' : formatInt(value);
      const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        "'": '&#39;',
        '"': '&quot;',
      }[char]));
      const percent = (account) => {
        if (!account.token_limit) return 0;
        if (!account.token_limit) return 0;
        return Math.min(100, Math.floor((Number(account.total_tokens || 0) / account.token_limit) * 100));
      };
      const formatReset = (resetAt) => {
        if (!resetAt) return 'Not set';
        const seconds = Math.max(0, Math.floor(resetAt - (Date.now() / 1000)));
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return hours ? `${hours}h ${minutes}m` : `${minutes}m`;
      };
      const timeAgo = (timestamp) => {
        if (!timestamp) return 'Never';
        const seconds = Math.max(0, Math.floor((Date.now() / 1000) - timestamp));
        if (seconds < 60) return 'Just now';
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes}m ago`;
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h ${minutes % 60}m ago`;
        return `${Math.floor(hours / 24)}d ago`;
      };
      const statusClass = (status) => {
        const code = Number(status || 0);
        if (code >= 200 && code < 300) return 'ok';
        if (code === 429) return 'warn';
        if (code >= 400) return 'error';
        return 'muted';
      };
      const renderRequestLog = (entries = []) => {
        if (!entries.length) {
          return '<div class="chart-empty">No proxied requests yet. Send traffic to <code>/v1</code> to populate this log.</div>';
        }
        return entries.map((entry) => {
          const model = entry.model ? `${escapeHtml(entry.model)}${entry.mapped_model ? ` → ${escapeHtml(entry.mapped_model)}` : ''}` : 'Unknown model';
          const attempts = (entry.attempts || []).map((attempt) => `${escapeHtml(attempt.account)}:${escapeHtml(attempt.status)}`).join(' · ');
          return `
            <div class="request-log-row">
              <div class="request-log-main">
                <span class="request-status ${statusClass(entry.status)}">${escapeHtml(entry.status)}</span>
                <div>
                  <b>${escapeHtml(entry.method || 'GET')} ${escapeHtml(entry.path || '/v1')}</b>
                  <span>${model}</span>
                </div>
              </div>
              <div class="request-log-meta">
                <span>${escapeHtml(entry.account || 'No account accepted')}</span>
                <span>${formatInt(entry.duration_ms)}ms</span>
                <span>${timeAgo(entry.timestamp)}</span>
              </div>
              <code>${attempts || 'No attempts recorded'}</code>
            </div>`;
        }).join('');
      };
      const accountCard = (account) => `
        <div class="account-card">
          <div class="account-top">
            <div>
              <b>${escapeHtml(account.name)}</b>
              <code>${escapeHtml(account.account_id)}</code>
            </div>
            <span class="account-status">Active</span>
          </div>
          <div class="account-total">
            <strong>${formatInt(account.total_tokens)}</strong>
            <span>observed tokens</span>
          </div>
          <div class="meter"><i style="width: ${percent(account)}%"></i></div>
          <div class="quota-grid">
            <span>Prompt <b>${formatInt(account.prompt_tokens)}</b></span>
            <span>Completion <b>${formatInt(account.completion_tokens)}</b></span>
            <span>Limit <b>${formatOptionalInt(account.token_limit)}</b></span>
            <span>Remaining <b>${formatOptionalInt(account.remaining_tokens)}</b></span>
            <span>Requests <b>${formatInt(account.requests)}</b></span>
            <span>Reset <b>${formatReset(account.reset_at)}</b></span>
          </div>
          <div class="account-foot">
            <span>Unknown-token responses: <b>${formatInt(account.unknown_token_responses)}</b></span>
            <span>Last used: <b>${timeAgo(account.last_used_at)}</b></span>
          </div>
        </div>`;
      const renderUsage = ({ accounts = [], request_log = [] }) => {
        const totalObserved = accounts.reduce((sum, account) => sum + Number(account.total_tokens || 0), 0);
        const totalLimit = accounts.reduce((sum, account) => sum + Number(account.token_limit || 0), 0);
        const totalRemaining = totalLimit ? Math.max(0, totalLimit - totalObserved) : null;
        const totalRequests = accounts.reduce((sum, account) => sum + Number(account.requests || 0), 0);
        const totalUnknown = accounts.reduce((sum, account) => sum + Number(account.unknown_token_responses || 0), 0);
        const configuredQuota = accounts.filter((account) => account.token_limit).length;

        $('quota-configured').textContent = `${configuredQuota}/${accounts.length} configured`;
        $('quota-observed').textContent = formatInt(totalObserved);
        const quotaPercent = totalLimit ? Math.min(100, Math.floor((totalObserved / totalLimit) * 100)) : 0;
        if ($('quota-meter')) $('quota-meter').style.width = `${quotaPercent}%`;
        $('quota-donut').style.strokeDasharray = `${quotaPercent} 100`;
        $('quota-percent').textContent = `${quotaPercent}%`;
        if ($('quota-remaining')) $('quota-remaining').textContent = formatOptionalInt(totalRemaining);
        $('quota-unknown').textContent = formatInt(totalUnknown);
        $('stat-accounts').textContent = formatInt(accounts.length);
        $('stat-requests').textContent = formatInt(totalRequests);
        $('stat-tokens').textContent = formatInt(totalObserved);
        $('stat-remaining').textContent = formatOptionalInt(totalRemaining);
        $('account-bars').innerHTML = renderBars(accounts);
        if ($('request-mix-bars')) $('request-mix-bars').innerHTML = renderRequestMix(totalRequests, totalUnknown);
        if ($('composition-chart')) $('composition-chart').innerHTML = renderComposition(accounts);
        if ($('quota-rings')) $('quota-rings').innerHTML = renderQuotaRings(accounts);
        if ($('request-log')) $('request-log').innerHTML = renderRequestLog(request_log);
        if ($('request-log-count')) $('request-log-count').textContent = `${request_log.length} recent`;
        if ($('accounts-list')) $('accounts-list').innerHTML = accounts.length ? accounts.map(accountCard).join('') : `
          <div class="empty-state">
            <b>No accounts configured</b>
            <p>Add Cloudflare account IDs and API tokens locally. The proxy will stay locked until setup is complete.</p>
            <a class="button" href="/setup">Open Setup</a>
          </div>`;
        $('usage-sync').textContent = `Backend synced ${new Date().toLocaleTimeString()}.`;
      };
      const renderBars = (accounts) => {
        if (!accounts.length) {
          return '<div class="chart-empty">Add accounts to see token distribution.</div>';
        }
        const maxTokens = Math.max(...accounts.map((account) => Number(account.total_tokens || 0)), 1);
        return accounts.map((account, index) => {
          const total = Number(account.total_tokens || 0);
          const width = Math.max(2, Math.round((total / maxTokens) * 100));
          return `
            <div style="margin-bottom: 8px;">
              <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; color: var(--text); font-weight: 500;">
                <b>${escapeHtml(account.name)}</b>
                <span style="color: var(--muted); font-family: var(--mono);">${formatInt(total)} tokens</span>
              </div>
              <div style="width: 100%; height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; position: relative;">
                <div style="width: ${width}%; height: 100%; background: var(--accent); border-radius: 4px; transition: width 0.3s ease;"></div>
              </div>
            </div>`;
        }).join('');
      };
      const renderRequestMix = (totalRequests, totalUnknown) => {
        const known = Math.max(0, totalRequests - totalUnknown);
        const max = Math.max(known, totalUnknown, 1);
        return [
          ['Known', known, 'fill-0'],
          ['Unknown', totalUnknown, 'fill-3'],
        ].map(([label, value, fill]) => `
          <div class="spark-row">
            <span>${label}</span>
            <i><b class="${fill}" style="height:${Math.max(8, Math.round((value / max) * 100))}%"></b></i>
            <strong>${formatInt(value)}</strong>
          </div>`).join('');
      };
      const renderComposition = (accounts) => {
        const prompt = accounts.reduce((sum, account) => sum + Number(account.prompt_tokens || 0), 0);
        const completion = accounts.reduce((sum, account) => sum + Number(account.completion_tokens || 0), 0);
        const total = Math.max(prompt + completion, 1);
        return `
          <div class="composition-stack">
            <i class="fill-1" style="width:${Math.max(2, Math.round((prompt / total) * 100))}%"></i>
            <i class="fill-2" style="width:${Math.max(2, Math.round((completion / total) * 100))}%"></i>
          </div>
          <div class="composition-meta">
            <span>Prompt <b>${formatInt(prompt)}</b></span>
            <span>Completion <b>${formatInt(completion)}</b></span>
          </div>`;
      };
      const renderQuotaRings = (accounts) => {
        if (!accounts.length) return '<div class="chart-empty">No account quota data yet.</div>';
        return accounts.slice(0, 6).map((account, index) => {
          const used = percent(account);
          return `
            <div class="ring-chip">
              <svg viewBox="0 0 42 42" aria-label="${escapeHtml(account.name)} quota">
                <circle class="ring-track" cx="21" cy="21" r="15"></circle>
                <circle class="ring-value fill-stroke-${index % 4}" cx="21" cy="21" r="15" style="stroke-dasharray:${used} 100"></circle>
              </svg>
              <span><b>${used}%</b>${escapeHtml(account.name)}</span>
            </div>`;
        }).join('');
      };
      const refreshUsage = async () => {
        try {
          const response = await fetch('/usage', { headers: { accept: 'application/json' } });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          renderUsage(await response.json());
        } catch (error) {
          $('usage-sync').textContent = `Backend sync failed: ${error.message}.`;
        }
      };

      document.querySelectorAll('[data-copy]').forEach((button) => {
        button.addEventListener('click', async () => {
          await navigator.clipboard.writeText(button.dataset.copy);
          button.textContent = 'Copied';
          setTimeout(() => button.textContent = 'Copy', 1200);
        });
      });
      if (themeButton) {
        themeButton.addEventListener('click', () => {
          applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
        });
      }
      
      const tabBtns = document.querySelectorAll('.tab-btn');
      const tabPanels = document.querySelectorAll('.tab-panel');
      
      const activateTab = (tabId) => {
        tabBtns.forEach(b => b.classList.remove('active'));
        tabPanels.forEach(p => p.style.display = 'none');
        const btn = Array.from(tabBtns).find(b => b.dataset.tab === tabId);
        if (btn) btn.classList.add('active');
        const panel = document.getElementById(tabId);
        if (panel) panel.style.display = 'block';
      };
      
      tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
          if (window.location.pathname !== '/') {
            window.location.href = '/';
            return;
          }
          activateTab(btn.dataset.tab);
        });
      });
      
      if (window.location.search.includes('saved=1') || window.location.search.includes('error=')) {
         activateTab('tab-settings');
         history.replaceState({}, document.title, window.location.pathname);
      }
      
      const list = document.getElementById('account-setup-list');
      const count = document.getElementById('account-count');
      const addAccountBtn = document.getElementById('add-account');
      if (addAccountBtn) {
        addAccountBtn.addEventListener('click', () => {
          const index = Number(count.value) + 1;
          count.value = String(index);
          list.insertAdjacentHTML('beforeend', `
            <details open>
              <summary>Cloudflare Account #${index}</summary>
              <div class="field-grid">
                <label>Name <input name="name_${index}" placeholder="e.g. user@gmail.com" autocomplete="off"></label>
                <label>Account ID <input name="account_id_${index}" autocomplete="off" placeholder="Cloudflare account ID"></label>
                <label>API Token <input name="api_token_${index}" type="password" autocomplete="off" placeholder="Cloudflare API token"></label>
              </div>
            </details>`);
        });
      }
      
      refreshUsage();
      setInterval(refreshUsage, 5000);
    </script>"""


def _account_bars(snapshots: list[tuple[CloudflareAccount, dict[str, object]]]) -> str:
    if not snapshots:
        return '<div class="chart-empty" style="color: var(--muted); font-size: 13px; text-align: center; padding: 20px;">No accounts active</div>'
    max_tokens = max((int(item["total_tokens"]) for _, item in snapshots), default=0) or 1
    bars_html = []
    for index, (account, item) in enumerate(snapshots):
        pct = max(2, int((int(item['total_tokens']) / max_tokens) * 100))
        bars_html.append(f'''
        <div style="margin-bottom: 8px;">
          <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; color: var(--text); font-weight: 500;">
            <b>{escape(account.name)}</b>
            <span style="color: var(--muted); font-family: var(--mono);">{_format_int(int(item['total_tokens']))} tokens</span>
          </div>
          <div style="width: 100%; height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; position: relative;">
            <div style="width: {pct}%; height: 100%; background: var(--accent); border-radius: 4px; transition: width 0.3s ease;"></div>
          </div>
        </div>
        ''')
    return "".join(bars_html)


def _request_mix_bars(total_requests: int, total_unknown: int) -> str:
    known = max(0, total_requests - total_unknown)
    maximum = max(known, total_unknown, 1)
    rows = [("Known", known, "fill-0"), ("Unknown", total_unknown, "fill-3")]
    return "".join(
        f"""
        <div class="spark-row">
          <span>{label}</span>
          <i><b class="{fill}" style="height:{max(8, int((value / maximum) * 100))}%"></b></i>
          <strong>{_format_int(value)}</strong>
        </div>
        """
        for label, value, fill in rows
    )


def _composition_chart(snapshots: list[tuple[CloudflareAccount, dict[str, object]]]) -> str:
    prompt = sum(int(item["prompt_tokens"]) for _, item in snapshots)
    completion = sum(int(item["completion_tokens"]) for _, item in snapshots)
    total = max(prompt + completion, 1)
    prompt_width = max(2, int((prompt / total) * 100))
    completion_width = max(2, int((completion / total) * 100))
    return f"""
    <div class="composition-stack">
      <i class="fill-1" style="width:{prompt_width}%"></i>
      <i class="fill-2" style="width:{completion_width}%"></i>
    </div>
    <div class="composition-meta">
      <span>Prompt <b>{_format_int(prompt)}</b></span>
      <span>Completion <b>{_format_int(completion)}</b></span>
    </div>
    """


def _quota_rings(snapshots: list[tuple[CloudflareAccount, dict[str, object]]]) -> str:
    if not snapshots:
        return '<div class="chart-empty">No account quota data yet.</div>'
    return "".join(
        f"""
        <div class="ring-chip">
          <svg viewBox="0 0 42 42" aria-label="{escape(account.name)} quota">
            <circle class="ring-track" cx="21" cy="21" r="15"></circle>
            <circle class="ring-value fill-stroke-{index % 4}" cx="21" cy="21" r="15" style="stroke-dasharray:{_usage_percent(account, item)} 100"></circle>
          </svg>
          <span><b>{_usage_percent(account, item)}%</b>{escape(account.name)}</span>
        </div>
        """
        for index, (account, item) in enumerate(snapshots[:6])
    )


def _account_setup_block(index: int, account: CloudflareAccount) -> str:
    return f"""
    <details {'open' if index == 1 else ''}>
      <summary>Cloudflare Account #{index}</summary>
      <div class="field-grid">
        <label>Name <input name="name_{index}" value="{escape(account.name)}" placeholder="e.g. user@gmail.com" autocomplete="off"></label>
        <label>Account ID <input name="account_id_{index}" value="{escape(account.account_id)}" autocomplete="off" placeholder="Cloudflare account ID"></label>
        <label>API Token <input name="api_token_{index}" value="{escape(account.api_token)}" type="password" autocomplete="off" placeholder="Cloudflare API token"></label>
      </div>
    </details>
    """


def _codex_shell(title: str, subtitle: str, body: str, shell_class: str = "compact-shell") -> str:
    shell = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
            :root {
      color-scheme: light;
      --bg: #fafafa;
      --bg-rail: #f4f4f5;
      --panel: #ffffff;
      --panel-raised: #ffffff;
      --line: #e4e4e7;
      --line-strong: #d4d4d8;
      --text: #09090b;
      --muted: #71717a;
      --quiet: #a1a1aa;
      --accent: #ff0063;
      --accent-ink: #ffffff;
      --warn: #ea580c;
      --danger: #ef4444;
      --success: #10b981;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    :root[data-theme="dark"] {
      color-scheme: dark;
      --bg: #09090b;
      --bg-rail: #18181b;
      --panel: #09090b;
      --panel-raised: #18181b;
      --line: #27272a;
      --line-strong: #3f3f46;
      --text: #fafafa;
      --muted: #a1a1aa;
      --quiet: #71717a;
      --accent: #ff0063;
      --accent-ink: #ffffff;
      --warn: #f97316;
      --danger: #ef4444;
      --success: #10b981;
    }
    * { box-sizing: border-box; }
    html, body { min-height: 100%; margin: 0; background: var(--bg); color: var(--text); }
    body { background: var(--bg); }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
    }
    .stat-card {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 12px;
      position: relative;
    }
    .stat-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
    }
    .stat-title {
      font-size: 13px;
      font-weight: 500;
      color: var(--muted);
    }
    .stat-value {
      font-size: 32px;
      font-weight: 700;
      line-height: 1;
      color: var(--text);
      font-family: var(--mono);
      margin-bottom: 4px;
    }
    .stat-icon {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .pink-icon { background: color-mix(in srgb, var(--accent) 10%, transparent); color: var(--accent); }
    .cyan-icon { background: color-mix(in srgb, var(--text) 5%, transparent); color: var(--text); }
    .stat-sparkline {
      width: 100%;
      opacity: 0.85;
      margin-top: auto;
    }

    body { font-family: var(--sans); font-size: 14px; line-height: 1.45; }
    body::before, body::after { content: none !important; display: none !important; }
    a { color: inherit; }
    code { font-family: var(--mono); }
    main { min-height: 100vh; padding: 32px 20px; background: var(--bg); }
    .wrap { width: min(1120px, 100%); margin: 0 auto; }
    .wrap.dashboard-shell { width: min(1120px, 100%); }
    .card, .dashboard-section, .chart-panel, .mini-chart-panel, .endpoint-panel, .quota-panel, .account-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 32px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .account-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06); border-color: var(--accent); }
    .dashboard-shell > .card { padding: 0; background: transparent; border: 0; border-radius: 0; box-shadow: none; }
    .brand { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; }
    .logo { width: 36px; height: 36px; display: grid; place-items: center; background: var(--panel); border: 0; border-radius: 10px; }
    .terminal { width: 14px; height: 10px; border: 1px solid var(--accent); border-radius: 3px; }
    h1, h2, h3, p { margin-top: 0; }
    h1 { margin-bottom: 2px; font-size: 24px; font-weight: 700; letter-spacing: -0.03em; }
    h2 { margin-bottom: 12px; font-size: 28px; font-weight: 700; letter-spacing: -0.03em; }
    h3 { margin-bottom: 0; font-size: 16px; font-weight: 600; letter-spacing: -0.02em; }
    .subtitle, .muted, .footnote { color: var(--muted); }
    .dashboard-shell .brand { display: none; }
    .top-nav {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 16px 32px;
      background: color-mix(in srgb, var(--bg) 80%, transparent);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .nav-brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .nav-brand b {
      font-size: 15px;
      font-weight: 650;
      letter-spacing: -0.01em;
    }
    .tab-pills {
      display: flex;
      align-items: center;
      background: var(--bg-rail);
      padding: 4px;
      border-radius: 999px;
      border: 0;
    }
    .tab-btn {
      background: transparent;
      border: none;
      border-radius: 999px;
      padding: 6px 16px;
      font: 500 13px/1 var(--sans);
      color: var(--muted);
      cursor: pointer;
      transition: all 150ms ease;
    }
    .tab-btn:hover {
      color: var(--text);
    }
    .tab-btn.active {
      background: var(--panel);
      color: var(--text);
      font-weight: 600;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .nav-right {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .dashboard-hero {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
      margin-bottom: 24px;
    }
    .dashboard-hero h2 { margin: 0 0 4px; font-size: 24px; font-weight: 700; letter-spacing: -0.03em; text-wrap: balance; }
    .dashboard-hero p { margin: 0; max-width: 72ch; color: var(--muted); }
    .hero-actions { display: flex; align-items: center; gap: 10px; }
    .status, .account-status {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      white-space: nowrap;
      border: 1px solid var(--success);
      border-radius: 999px;
      padding: 5px 10px;
      color: var(--success);
      background: color-mix(in oklch, var(--success) 14%, var(--panel));
      font-size: 12px;
      font-weight: 650;
    }
    .status::before, .account-status::before { content: ""; width: 6px; height: 6px; border-radius: 999px; background: currentColor; }
    .status-warn { color: var(--warn); background: color-mix(in oklch, var(--warn) 14%, var(--panel)); border-color: color-mix(in oklch, var(--warn), var(--line) 40%); }
    .dashboard-section, .stats, .chart-grid, .mini-chart-grid { margin-bottom: 16px; }
    .section-title { display: flex; justify-content: space-between; align-items: baseline; gap: 14px; margin-bottom: 16px; }
    .section-title span { color: var(--quiet); font-size: 12px; }
    .accounts { display: grid; grid-template-columns: repeat(auto-fit, minmax(315px, 1fr)); gap: 12px; }
    .account-card { padding: 16px; background: var(--bg-rail); }
    .account-top { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 16px; }
    .account-top b { display: block; margin-bottom: 2px; font-size: 14px; }
    .account-top code { color: var(--quiet); font-size: 12px; }
    .account-total, .quota-total { display: flex; flex-direction: column; gap: 2px; margin-bottom: 14px; }
    .account-total strong, .quota-total strong, .stats b, .donut-center b {
      font-family: var(--mono);
      font-weight: 650;
      letter-spacing: -0.04em;
      color: var(--text);
    }
    .account-total strong { font-size: 28px; line-height: 1; }
    .quota-total strong { font-size: 34px; line-height: 1; }
    .account-total span, .quota-total span { color: var(--muted); font-size: 12px; }
    .quota-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1px; overflow: hidden; margin: 14px 0; border: 0; border-radius: 10px; background: var(--line); }
    .quota-grid span { display: flex; flex-direction: column; gap: 3px; min-width: 0; padding: 10px; background: var(--panel); color: var(--quiet); font-size: 11px; }
    .quota-grid b { overflow: hidden; color: var(--text); font-family: var(--mono); font-size: 12px; font-weight: 600; text-overflow: ellipsis; }
    .account-foot { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 14px; color: var(--quiet); font-size: 12px; }
    .account-foot b { color: var(--muted); font-weight: 600; }
    .meter { height: 8px; overflow: hidden; border-radius: 999px; background: var(--bg); border: 0; }
    .meter i { display: block; height: 100%; border-radius: inherit; background: var(--accent); transition: width 180ms ease-out; }
    .meter.large { height: 10px; margin-top: 14px; }

    .chart-grid { display: grid; grid-template-columns: minmax(280px, 0.8fr) minmax(0, 1.2fr); gap: 12px; }
    .mini-chart-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }
    .donut-wrap { position: relative; display: grid; place-items: center; min-height: 198px; }
    .donut { width: min(178px, 100%); transform: rotate(-90deg); overflow: visible; }
    .donut-track { stroke: var(--bg); stroke-width: 14; fill: none; }
    .donut-value { stroke: var(--accent); stroke-width: 14; fill: none; stroke-linecap: round; transition: stroke-dasharray 180ms ease-out; }
    .donut-center { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .donut-center b { font-size: 32px; }
    .donut-center span { color: var(--quiet); font-size: 12px; }
    .chart-legend { display: flex; justify-content: center; gap: 16px; color: var(--muted); font-size: 12px; }
    .chart-legend span { display: inline-flex; align-items: center; gap: 7px; }
    .chart-legend i { width: 8px; height: 8px; border-radius: 999px; }
    .legend-used { background: var(--accent); }
    .legend-left { background: var(--line-strong); }
    .bar-chart { display: flex; flex-direction: column; gap: 14px; }
    .bar-row { display: grid; gap: 7px; }
    .bar-label { display: flex; justify-content: space-between; gap: 12px; color: var(--muted); font-size: 12px; }
    .bar-label b { color: var(--text); font-weight: 600; }
    .bar-track { height: 9px; overflow: hidden; border-radius: 999px; background: var(--bg); border: 0; }
    .bar-fill { display: block; height: 100%; border-radius: inherit; }
    .request-log { display: grid; gap: 10px; max-height: 360px; overflow: auto; padding-right: 4px; }
    .request-log-row { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(220px, 0.8fr); gap: 10px 18px; align-items: center; padding: 12px; border-radius: 12px; background: var(--bg-rail); border: 1px solid var(--line); }
    .request-log-main { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .request-log-main div { min-width: 0; }
    .request-log-main b { display: block; overflow: hidden; color: var(--text); font-size: 13px; font-weight: 650; text-overflow: ellipsis; white-space: nowrap; }
    .request-log-main span:not(.request-status), .request-log-meta { color: var(--muted); font-size: 12px; }
    .request-log-meta { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px 12px; }
    .request-log-row code { grid-column: 1 / -1; overflow: auto; color: var(--quiet); font-size: 11px; white-space: nowrap; }
    .request-status { display: inline-flex; align-items: center; justify-content: center; min-width: 46px; border-radius: 999px; padding: 5px 8px; background: var(--panel); color: var(--muted); font-family: var(--mono); font-size: 12px; font-weight: 700; }
    .request-status.ok { color: var(--success); background: color-mix(in oklch, var(--success) 12%, var(--panel)); }
    .request-status.warn { color: var(--warn); background: color-mix(in oklch, var(--warn) 14%, var(--panel)); }
    .request-status.error { color: var(--danger); background: color-mix(in oklch, var(--danger) 14%, var(--panel)); }
    .fill-0 { background: var(--text); }
    .fill-1 { background: var(--muted); }
    .fill-2 { background: var(--text); }
    .fill-3 { background: var(--muted); }
    .spark-bars { display: flex; align-items: flex-end; justify-content: center; gap: 28px; min-height: 128px; }
    .spark-row { display: grid; grid-template-rows: auto 1fr auto; align-items: end; justify-items: center; gap: 8px; color: var(--muted); font-size: 12px; }
    .spark-row i { width: 28px; height: 88px; display: flex; align-items: flex-end; padding: 2px; background: var(--bg); border: 0; border-radius: 8px; }
    .spark-row b { width: 100%; border-radius: 5px; }
    .spark-row strong { color: var(--text); font-family: var(--mono); font-size: 12px; }
    .composition-stack { display: flex; height: 12px; overflow: hidden; border-radius: 999px; background: var(--bg); border: 0; }
    .composition-stack i { display: block; height: 100%; }
    .composition-meta, .endpoint-meta { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
    .composition-meta span, .endpoint-meta span { flex: 1 1 130px; min-width: 0; padding: 10px 12px; background: var(--bg-rail); border: 0; border-radius: 10px; color: var(--quiet); font-size: 12px; }
    .composition-meta b, .endpoint-meta b { display: block; overflow: hidden; margin-top: 2px; color: var(--text); font-family: var(--mono); font-size: 12px; font-weight: 600; text-overflow: ellipsis; }
    .radar-rings { display: flex; flex-wrap: wrap; gap: 10px; }
    .ring-chip { display: flex; align-items: center; gap: 10px; min-width: 132px; padding: 10px; background: var(--bg-rail); border: 0; border-radius: 10px; }
    .ring-chip svg { width: 42px; height: 42px; flex: 0 0 auto; transform: rotate(-90deg); }
    .ring-track { stroke: var(--bg); stroke-width: 6; fill: none; }
    .ring-value { stroke-width: 6; fill: none; stroke-linecap: round; }
    .fill-stroke-0 { stroke: var(--text); }
    .fill-stroke-1 { stroke: var(--muted); }
    .fill-stroke-2 { stroke: var(--text); }
    .fill-stroke-3 { stroke: var(--muted); }
    .ring-chip span { min-width: 0; color: var(--quiet); font-size: 12px; }
    .ring-chip b { display: block; color: var(--text); font-family: var(--mono); font-size: 13px; }
    .endpoint-panel label { display: block; color: var(--muted); font-size: 12px; }
    .copyline { display: flex; align-items: center; gap: 8px; margin-top: 8px; padding: 6px; background: var(--bg); border: 0; border-radius: 10px; }
    .copyline code { flex: 1; min-width: 0; overflow: auto; padding: 0 8px; color: var(--text); font-size: 13px; white-space: nowrap; }
    .button, .copyline button, button[type="submit"], .theme-toggle { display: inline-flex; align-items: center; justify-content: center; min-height: 36px; border: 1px solid color-mix(in oklch, var(--accent), var(--line) 30%); border-radius: 9px; padding: 8px 13px; background: var(--accent); color: var(--accent-ink); font: 650 13px/1 var(--sans); text-decoration: none; cursor: pointer; transition: background-color 160ms ease-out, border-color 160ms ease-out, transform 160ms ease-out; }
    .button:hover, .copyline button:hover, button[type="submit"]:hover, .theme-toggle:hover { background: color-mix(in oklch, var(--accent), white 14%); }
    .button:active, .copyline button:active, button[type="submit"]:active, .theme-toggle:active { transform: translateY(1px); }
    .button.secondary { background: var(--panel-raised); color: var(--text); border-color: var(--line-strong); }
    .button.secondary:hover { background: var(--bg-rail); }
    .theme-toggle { background: var(--panel); color: var(--text); border-color: var(--line-strong); }
    .info-strip { display: flex; justify-content: space-between; align-items: center; gap: 18px; padding: 16px 0 0; color: var(--muted); }
    .info-strip p { max-width: 76ch; margin: 0; font-size: 12px; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; }
    .empty-state, .chart-empty { padding: 24px; border: 1px dashed var(--line-strong); border-radius: 12px; background: var(--bg-rail); color: var(--muted); }
    .empty-state b { display: block; margin-bottom: 6px; color: var(--text); }
    .empty-state p { margin-bottom: 16px; }
    .stack { display: grid; gap: 14px; }
    #account-list { display: grid; gap: 12px; }
    details { background: var(--bg-rail); border: 0; border-radius: 12px; padding: 14px; }
    summary { cursor: pointer; font-weight: 650; }
    .field-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }
    label { color: var(--muted); font-size: 12px; }
    input { width: 100%; margin-top: 6px; border: 0; border-radius: 9px; padding: 10px 11px; background: var(--bg); color: var(--text); font: 13px/1.2 var(--sans); }
    input::placeholder { color: oklch(0.62 0.018 255); }
    .notice { margin: 14px 0; padding: 12px; border-radius: 10px; border: 0; }
    .notice.success { color: var(--success); background: oklch(0.24 0.035 152 / 0.5); border-color: oklch(0.48 0.08 152); }
    .notice.error { color: var(--danger); background: oklch(0.24 0.035 24 / 0.5); border-color: oklch(0.48 0.1 24); }
    :focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
    @media (max-width: 920px) {
      main { padding: 14px; }
      .dashboard-hero, .info-strip { flex-direction: column; align-items: stretch; }
      .stats, .chart-grid, .mini-chart-grid { grid-template-columns: 1fr; }
      .request-log-row { grid-template-columns: 1fr; }
      .request-log-meta { justify-content: flex-start; }
      .quota-grid, .field-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 560px) {
      main { padding: 12px; }
      .card, .dashboard-section, .chart-panel, .mini-chart-panel, .endpoint-panel, .quota-panel { padding: 16px; }
      .accounts { grid-template-columns: 1fr; }
      .quota-grid, .field-grid { grid-template-columns: 1fr; }
      .copyline { align-items: stretch; flex-direction: column; }
      .copyline code { width: 100%; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { scroll-behavior: auto !important; transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; }
    }

  </style>
</head>
<body>
  <nav class="top-nav">
    <div class="nav-brand">
      <img src="/logo.png" style="height: 64px; object-fit: contain;" alt="Logo">
      <b>__TITLE__</b>
    </div>
    <div class="tab-pills">
      <button type="button" class="tab-btn active" data-tab="tab-dashboard">Dashboard</button>
      <button type="button" class="tab-btn" data-tab="tab-accounts">Accounts</button>
      <button type="button" class="tab-btn" data-tab="tab-settings">Settings</button>
    </div>
    <div class="nav-right">
      <button class="theme-toggle" type="button" id="theme-toggle" aria-label="Switch color theme">Dark theme</button>
    </div>
  </nav>
  <main>
    <div class="wrap __SHELL_CLASS__">
      <div class="card">__BODY__</div>
    </div>
  </main>
</body>
</html>"""
    return (
        shell.replace("__TITLE__", escape(title))
        .replace("__SUBTITLE__", escape(subtitle))
        .replace("__SHELL_CLASS__", escape(shell_class))
        .replace("__BODY__", body)
    )


def _form_value(form: dict[str, list[str]], key: str) -> str:
    values = form.get(key) or [""]
    return values[0].strip()


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_int(value: int) -> str:
    return f"{value:,}"


def _format_optional_int(value: object) -> str:
    if value is None:
        return "Not set"
    return _format_int(int(value))


def _format_reset(reset_at: object) -> str:
    if reset_at is None:
        return "Not set"
    seconds = max(0, int(float(reset_at) - time.time()))
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _format_time_ago(timestamp: object) -> str:
    if timestamp is None:
        return "Never"
    seconds = max(0, int(time.time() - float(timestamp)))
    if seconds < 60:
        return "Just now"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m ago"
    days = hours // 24
    return f"{days}d ago"


def _usage_percent(account: CloudflareAccount, item: dict[str, object]) -> int:
    if not account.token_limit:
        return 0
    return min(100, int((int(item["total_tokens"]) / account.token_limit) * 100))


def _mask_account_id(account_id: str) -> str:
    if len(account_id) <= 8:
        return account_id
    return f"{account_id[:4]}...{account_id[-4:]}"


def main() -> None:
    settings = load_settings()
    uvicorn.run("pocket_lb.proxy:create_app", host=settings.host, port=settings.port, factory=True)


if __name__ == "__main__":
    main()
