//! ferralloy-fleet — the open, self-hostable fleet plane.
//!
//! Every serious edge platform ships an open device agent and gates the fleet
//! server behind a paid cloud. Ferralloy's fleet plane is open too, with no
//! feature gate — that is the whole point. It is deliberately small:
//!
//!   • **channels** (stable, beta, …) each hold one current **release** — a
//!     signed `.fpack`, validated on upload (signature + digests) so a broken
//!     or unsigned artifact can never become a channel target.
//!   • **devices** poll their channel, pull the pack, and run it through their
//!     OWN accept gate (behavioral verification on-device) before it goes live,
//!     then **report** the outcome here. The server never pushes; a device
//!     behind NAT still updates, and "rolled out" means "behavior verified on
//!     the device," not "bytes delivered."
//!
//! State is a directory: `channels/<name>.fpack` + a small `channels.json` and
//! `devices.json`. No database, no cloud — copy the directory to move the
//! fleet. This is the reference server; a production one would add cohorts,
//! staged canaries, and TUF-rooted keys (all planned, all open).

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Json, Response};
use axum::routing::{get, put};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone, Serialize, Deserialize)]
struct Release {
    name: String,
    version: String,
    sha256: String,
    signer: String,
    /// seconds since epoch, stamped by the server on upload
    published: u64,
}

// `device` and `seen` are stamped by the server, so a device's POST omits
// them; `#[serde(default)]` on every field also makes the report robust to a
// client that sends a subset.
#[derive(Clone, Serialize, Deserialize, Default)]
#[serde(default)]
struct DeviceReport {
    device: String,
    channel: String,
    target_sha: String,
    version: String,
    behavior: String,
    ok: bool,
    platform: String,
    /// seconds since epoch of the last report
    seen: u64,
}

/// A staged (canary) rollout: a new release going to a fraction of a channel's
/// devices while the rest stay on `current`. Promotion is an operator decision,
/// ideally made from the canary's VERIFIED-behavior pass rate (the dashboard).
#[derive(Clone, Serialize, Deserialize)]
struct Rollout {
    release: Release,
    /// 0..=100 — a device runs the rollout iff its stable percentile < percent.
    percent: u8,
}

/// A channel's state: the current release plus an optional in-flight rollout.
#[derive(Clone, Serialize, Deserialize)]
struct Channel {
    current: Release,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rollout: Option<Rollout>,
}

#[derive(Default, Serialize, Deserialize)]
struct FleetState {
    channels: BTreeMap<String, Channel>,
    devices: BTreeMap<String, DeviceReport>,
}

/// Deterministic 0..=99 bucket for a device id — FNV-1a mod 100. Stable across
/// polls and servers, so a device stays in or out of a canary consistently.
fn pctl(device: &str) -> u8 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in device.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    (h % 100) as u8
}

/// Which release this device should run on the channel, honoring the canary.
fn resolve<'a>(ch: &'a Channel, device: &str) -> &'a Release {
    match &ch.rollout {
        Some(r) if pctl(device) < r.percent => &r.release,
        _ => &ch.current,
    }
}

#[derive(Clone)]
struct App {
    inner: Arc<Mutex<FleetState>>,
    root: Arc<PathBuf>,
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let port: u16 = std::env::var("FERRALLOY_FLEET_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(7280);
    let root = std::env::var("FERRALLOY_FLEET_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::home_dir()
                .expect("no home")
                .join(".ferralloy")
                .join("fleet")
        });
    fs::create_dir_all(root.join("channels"))?;

    let mut state = FleetState::default();
    if let Ok(bytes) = fs::read(root.join("channels.json")) {
        state.channels = serde_json::from_slice(&bytes).unwrap_or_default();
    }
    if let Ok(bytes) = fs::read(root.join("devices.json")) {
        state.devices = serde_json::from_slice(&bytes).unwrap_or_default();
    }

    let app = App {
        inner: Arc::new(Mutex::new(state)),
        root: Arc::new(root),
    };

    let router = axum::Router::new()
        .route("/", get(dashboard))
        .route("/v1/fleet", get(fleet_json))
        .route("/v1/channels/{ch}", put(set_channel).get(get_channel))
        .route("/v1/channels/{ch}/pack", get(get_pack))
        .route("/v1/channels/{ch}/promote", axum::routing::post(promote))
        .route("/v1/channels/{ch}/abort", axum::routing::post(abort))
        .route("/v1/devices/{id}/report", axum::routing::post(post_report))
        .with_state(app)
        .layer(axum::extract::DefaultBodyLimit::max(512 * 1024 * 1024));

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    println!("ferralloy-fleet listening on 0.0.0.0:{port}");
    axum::serve(listener, router).await?;
    Ok(())
}

fn persist(app: &App, st: &FleetState) {
    let _ = fs::write(
        app.root.join("channels.json"),
        serde_json::to_vec_pretty(&st.channels).unwrap_or_default(),
    );
    let _ = fs::write(
        app.root.join("devices.json"),
        serde_json::to_vec_pretty(&st.devices).unwrap_or_default(),
    );
}

/// PUT a `.fpack` as a channel's release. Statically verified (signature +
/// digests) before it can become a target — the fleet plane never serves an
/// artifact it hasn't validated. `?canary=<N>` stages it as a rollout to N% of
/// the channel's devices instead of replacing `current`; without it, the
/// release becomes `current` and any in-flight rollout is cleared.
async fn set_channel(
    State(app): State<App>,
    Path(ch): Path<String>,
    Query(q): Query<std::collections::HashMap<String, String>>,
    body: axum::body::Bytes,
) -> Response {
    if ch.contains(['/', '.']) {
        return (StatusCode::BAD_REQUEST, "bad channel name").into_response();
    }
    let canary: Option<u8> = q.get("canary").and_then(|s| s.parse().ok()).map(|p: u8| p.min(100));
    let tmp = app.root.join("channels").join(format!("{ch}.incoming"));
    if let Err(e) = fs::write(&tmp, &body) {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("io: {e}")).into_response();
    }
    let pack = match ferralloy_pack::load(&tmp) {
        Ok(p) => p,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("load: {e}")).into_response(),
    };
    let signer = match ferralloy_pack::verify(&pack) {
        Ok(s) => s,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("verification failed: {e}")).into_response(),
    };
    let rel = Release {
        name: pack.manifest.name.clone(),
        version: pack.manifest.version.clone(),
        sha256: ferralloy_pack::sha256_hex(&body),
        signer,
        published: now(),
    };
    // canary pack stored alongside current so both can be served per-device
    let slot = if canary.is_some() { format!("{ch}.rollout.fpack") } else { format!("{ch}.fpack") };
    if let Err(e) = fs::rename(&tmp, app.root.join("channels").join(&slot)) {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("store: {e}")).into_response();
    }
    let mut st = app.inner.lock().unwrap();
    let msg = match canary {
        Some(p) => {
            match st.channels.get_mut(&ch) {
                Some(c) => { c.rollout = Some(Rollout { release: rel.clone(), percent: p }); }
                None => return (StatusCode::BAD_REQUEST, "cannot canary an empty channel — set a current release first").into_response(),
            }
            serde_json::json!({ "channel": ch, "canary": rel, "percent": p })
        }
        None => {
            st.channels.insert(ch.clone(), Channel { current: rel.clone(), rollout: None });
            let _ = fs::remove_file(app.root.join("channels").join(format!("{ch}.rollout.fpack")));
            serde_json::json!({ "channel": ch, "released": rel })
        }
    };
    persist(&app, &st);
    (StatusCode::OK, Json(msg)).into_response()
}

/// Promote the in-flight rollout to `current` (the canary becomes the fleet-
/// wide release). Operator action — run it once the canary's pass rate is good.
async fn promote(State(app): State<App>, Path(ch): Path<String>) -> Response {
    let mut st = app.inner.lock().unwrap();
    let Some(c) = st.channels.get_mut(&ch) else {
        return (StatusCode::NOT_FOUND, "no such channel").into_response();
    };
    let Some(r) = c.rollout.take() else {
        return (StatusCode::BAD_REQUEST, "no rollout to promote").into_response();
    };
    c.current = r.release.clone();
    let _ = fs::rename(
        app.root.join("channels").join(format!("{ch}.rollout.fpack")),
        app.root.join("channels").join(format!("{ch}.fpack")),
    );
    persist(&app, &st);
    (StatusCode::OK, Json(serde_json::json!({ "channel": ch, "promoted": r.release }))).into_response()
}

/// Abort the in-flight rollout — canary devices revert to `current` next poll.
async fn abort(State(app): State<App>, Path(ch): Path<String>) -> Response {
    let mut st = app.inner.lock().unwrap();
    let Some(c) = st.channels.get_mut(&ch) else {
        return (StatusCode::NOT_FOUND, "no such channel").into_response();
    };
    let had = c.rollout.take().is_some();
    let _ = fs::remove_file(app.root.join("channels").join(format!("{ch}.rollout.fpack")));
    persist(&app, &st);
    (StatusCode::OK, Json(serde_json::json!({ "channel": ch, "aborted": had }))).into_response()
}

async fn get_channel(
    State(app): State<App>,
    Path(ch): Path<String>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> Response {
    let st = app.inner.lock().unwrap();
    let Some(c) = st.channels.get(&ch) else {
        return (StatusCode::NOT_FOUND, "no release on this channel").into_response();
    };
    // With ?device, resolve to that device's release (honors the canary);
    // without it, report the full channel state (operator view).
    match q.get("device") {
        Some(dev) => Json(resolve(c, dev)).into_response(),
        None => Json(c).into_response(),
    }
}

async fn get_pack(
    State(app): State<App>,
    Path(ch): Path<String>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> Response {
    if ch.contains(['/', '.']) {
        return (StatusCode::BAD_REQUEST, "bad channel name").into_response();
    }
    // Serve the canary pack only to devices the canary covers.
    let on_canary = {
        let st = app.inner.lock().unwrap();
        match (q.get("device"), st.channels.get(&ch)) {
            (Some(dev), Some(c)) => c.rollout.as_ref().is_some_and(|r| pctl(dev) < r.percent),
            _ => false,
        }
    };
    let slot = if on_canary { format!("{ch}.rollout.fpack") } else { format!("{ch}.fpack") };
    match fs::read(app.root.join("channels").join(&slot)) {
        Ok(bytes) => (
            [(axum::http::header::CONTENT_TYPE, "application/vnd.ferralloy.fpack")],
            bytes,
        )
            .into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "no pack").into_response(),
    }
}

async fn post_report(State(app): State<App>, Path(id): Path<String>, Json(mut rep): Json<DeviceReport>) -> Response {
    rep.device = id.clone();
    rep.seen = now();
    let mut st = app.inner.lock().unwrap();
    st.devices.insert(id, rep);
    persist(&app, &st);
    (StatusCode::OK, Json(serde_json::json!({ "ok": true }))).into_response()
}

async fn fleet_json(State(app): State<App>) -> Json<serde_json::Value> {
    let st = app.inner.lock().unwrap();
    Json(serde_json::json!({ "channels": st.channels, "devices": st.devices }))
}

async fn dashboard(State(app): State<App>) -> Html<String> {
    let st = app.inner.lock().unwrap();
    let t = now();
    let mut chan_rows = String::new();
    for (name, c) in &st.channels {
        let r = &c.current;
        chan_rows.push_str(&format!(
            "<tr><td><b>{name}</b></td><td>{}</td><td class=m>{}…</td><td colspan=2 class=dim>current</td></tr>",
            r.version, &r.sha256[..r.sha256.len().min(12)]
        ));
        if let Some(ro) = &c.rollout {
            // canary health from device reports: of devices whose target_sha is
            // the canary's, how many verified? This is the promotion signal.
            let on: Vec<_> = st.devices.values()
                .filter(|d| d.channel == *name && d.target_sha == ro.release.sha256).collect();
            let ok = on.iter().filter(|d| d.ok).count();
            let health = if on.is_empty() { "no reports yet".to_string() }
                else { format!("{ok}/{} verified", on.len()) };
            let cls = if !on.is_empty() && ok == on.len() { "ok" } else if on.iter().any(|d| !d.ok) { "bad" } else { "dim" };
            chan_rows.push_str(&format!(
                "<tr><td class=dim>└ canary</td><td>{}</td><td class=m>{}…</td><td>{}%</td><td class={cls}>{health}</td></tr>",
                ro.release.version, &ro.release.sha256[..ro.release.sha256.len().min(12)], ro.percent
            ));
        }
    }
    if chan_rows.is_empty() {
        chan_rows = "<tr><td colspan=5 class=dim>no channels yet — <span class=m>ferralloy release &lt;pack&gt; --channel stable --fleet …</span></td></tr>".into();
    }
    let mut dev_rows = String::new();
    for (id, d) in &st.devices {
        let age = t.saturating_sub(d.seen);
        let dot = if d.ok { "ok" } else { "bad" };
        dev_rows.push_str(&format!(
            "<tr><td><b>{id}</b></td><td class=m>{}</td><td>{}</td><td>{}</td><td class={dot}>{}</td><td class=dim>{age}s ago</td></tr>",
            d.platform, d.channel, d.version, d.behavior
        ));
    }
    if dev_rows.is_empty() {
        dev_rows = "<tr><td colspan=6 class=dim>no devices reporting — start ferralloyd with FERRALLOY_FLEET_URL set</td></tr>".into();
    }
    Html(format!(
        r#"<!doctype html><meta charset=utf-8><title>Ferralloy Fleet</title>
<style>
 body{{font:14px ui-monospace,Menlo,monospace;background:#0b1024;color:#eef1f8;margin:0;padding:28px 34px}}
 h1{{font-size:17px;letter-spacing:.02em;margin:0 0 2px}} h1 b{{color:#cfaa5b}}
 .sub{{color:#8aa0bd;font-size:12px;margin:0 0 22px}}
 h2{{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#cfaa5b;margin:26px 0 8px}}
 table{{width:100%;border-collapse:collapse}} td,th{{text-align:left;padding:8px 12px;border-bottom:1px solid #243157;font-size:12.5px}}
 th{{color:#8aa0bd;font-size:10px;letter-spacing:.1em;text-transform:uppercase}}
 .m{{color:#9fb2cc}} .dim{{color:#61708a}} .ok{{color:#46c6b0}} .bad{{color:#ff6a6a}}
</style>
<h1><b>Φ</b> Ferralloy Fleet</h1>
<p class=sub>Open fleet plane · a device shows here only after it VERIFIES a release's behavior on-device. Rolled out = verified, not delivered.</p>
<h2>Channels</h2>
<table><tr><th>channel</th><th>version</th><th>sha256</th><th>rollout</th><th>canary health</th></tr>{chan_rows}</table>
<h2>Devices</h2>
<table><tr><th>device</th><th>platform</th><th>channel</th><th>version</th><th>behavior</th><th>last seen</th></tr>{dev_rows}</table>
"#
    ))
}
