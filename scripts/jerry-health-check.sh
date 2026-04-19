#!/usr/bin/env bash
# jerry-health-check.sh
# Pings Jerry's /health endpoint every CHECK_INTERVAL_SECONDS (default 300).
# If Jerry does not return HTTP 200 with status:ok within TIMEOUT_SECONDS,
# sends a Discord alert.  Alerts are rate-limited: only one alert per
# ALERT_COOLDOWN_SECONDS (default 1800 = 30 min) regardless of how long
# the outage lasts, plus a RECOVERED notification when it comes back up.
#
# Environment variables (all optional — defaults shown below):
#   DISCORD_WEBHOOK_URL       — where to send alerts (required for notifications)
#   JERRY_URL                 — default http://127.0.0.1:8088
#   CHECK_INTERVAL_SECONDS    — how often to poll (default 300)
#   TIMEOUT_SECONDS           — curl timeout per probe (default 5)
#   ALERT_COOLDOWN_SECONDS    — minimum seconds between DOWN alerts (default 1800)
#
# Start/stop via LaunchAgent:
#   ~/Library/LaunchAgents/site.tca-infraforge.jerry-health-check.plist
#   launchctl bootout  gui/$(id -u) ~/Library/LaunchAgents/site.tca-infraforge.jerry-health-check.plist
#   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/site.tca-infraforge.jerry-health-check.plist
# --------------------------------------------------------------------------

set -euo pipefail

JERRY_URL="${JERRY_URL:-http://127.0.0.1:8088}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-5}"
ALERT_COOLDOWN_SECONDS="${ALERT_COOLDOWN_SECONDS:-1800}"
DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL:-}"

HEALTH_URL="${JERRY_URL}/health"
_last_alert_ts=0
_was_down=false

log() {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [jerry-health] $*"
}

discord_send() {
    local payload="$1"
    if [[ -z "$DISCORD_WEBHOOK_URL" ]]; then
        log "DISCORD_WEBHOOK_URL not set — notification skipped"
        return
    fi
    curl -sSf -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$DISCORD_WEBHOOK_URL" \
        --max-time 10 \
        --retry 2 \
        --retry-delay 3 \
        -o /dev/null 2>&1 || log "Discord POST failed (non-fatal)"
}

alert_down() {
    local now
    now=$(date +%s)
    local elapsed=$(( now - _last_alert_ts ))
    if (( elapsed < ALERT_COOLDOWN_SECONDS )); then
        log "DOWN alert suppressed (cooldown — ${elapsed}s < ${ALERT_COOLDOWN_SECONDS}s)"
        return
    fi
    _last_alert_ts=$now
    local ts
    ts=$(date -u +"%Y-%m-%d %H:%M UTC")
    local payload
    payload=$(cat <<EOF
{
  "embeds": [{
    "title": "🔴 Jerry is DOWN",
    "description": "Jerry did not respond to /health within ${TIMEOUT_SECONDS}s.",
    "color": 15158332,
    "fields": [
      {"name": "URL", "value": "\`${HEALTH_URL}\`", "inline": true},
      {"name": "Host", "value": "\`$(hostname -s)\`", "inline": true}
    ],
    "footer": {"text": "Jerry self-health • ${ts}"},
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  }]
}
EOF
)
    log "Sending DOWN alert to Discord"
    discord_send "$payload"
}

alert_recovered() {
    local ts
    ts=$(date -u +"%Y-%m-%d %H:%M UTC")
    local payload
    payload=$(cat <<EOF
{
  "embeds": [{
    "title": "✅ Jerry RECOVERED",
    "description": "Jerry is responding to /health normally.",
    "color": 3066993,
    "fields": [
      {"name": "URL", "value": "\`${HEALTH_URL}\`", "inline": true}
    ],
    "footer": {"text": "Jerry self-health • ${ts}"},
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  }]
}
EOF
)
    log "Sending RECOVERED alert to Discord"
    discord_send "$payload"
}

check_once() {
    local http_code status
    # Use curl: get HTTP code and response body.  Timeout hard at TIMEOUT_SECONDS.
    local body
    body=$(curl -sf --max-time "$TIMEOUT_SECONDS" "$HEALTH_URL" 2>/dev/null) || {
        log "FAIL — curl returned non-zero (timeout or connection refused)"
        if [[ "$_was_down" == "false" ]]; then
            _was_down=true
        fi
        alert_down
        return
    }

    # Check JSON status field
    status=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "parse_error")
    if [[ "$status" == "ok" || "$status" == "degraded" ]]; then
        if [[ "$_was_down" == "true" ]]; then
            log "RECOVERED — status=${status}"
            _was_down=false
            _last_alert_ts=0   # reset cooldown so next outage alerts immediately
            alert_recovered
        else
            log "OK — status=${status}"
        fi
    else
        log "FAIL — unexpected status: ${status}"
        if [[ "$_was_down" == "false" ]]; then
            _was_down=true
        fi
        alert_down
    fi
}

log "Starting Jerry health monitor — polling ${HEALTH_URL} every ${CHECK_INTERVAL_SECONDS}s"

while true; do
    check_once
    sleep "$CHECK_INTERVAL_SECONDS"
done
