//! The **certificate** facet — verified *correctness*, the axis beyond verified *behavior*.
//!
//! Ferralloy's eval vectors prove a pack *reproduces* signed outputs bit-for-bit across fabrics.
//! A certificate proves something stronger and different: that the deployed control energy still
//! carries a valid formal **Lyapunov** guarantee — it drives the body into its basin and keeps it
//! there — re-proven ON THE DEVICE before the pack is trusted, with no SDP/SMT solver and nothing
//! from libm beyond `tanh`. The whole check is arithmetic + a box worklist, so it runs unchanged
//! from a Jetson to a browser (wasm32).
//!
//! This is the device-side re-verifier from the Charlot Lab certificate program (the 2nd-order
//! Taylor model + per-box CROWN |tanh″| bound, adaptive box refinement) generalized to read its
//! weights from the manifest instead of embedding them. Synthesis lineage: Chang/Gao (Neural
//! Lyapunov Control, dReal); our legs are the ternary weights, the quadratic-anchored init, and
//! train-stricter-than-verify margins. SOS/dReal stay a build-time/fleet gate — they need solvers;
//! this Taylor+CROWN pass is the on-device gate.

use serde::{Deserialize, Serialize};

/// A learned ternary Lyapunov energy: V(e) = eᵀPe + Σⱼ w₂ⱼ·tanh(s·(Tⱼ·e)+b₁ⱼ) − v₀,
/// with T ∈ {−1,0,+1} (the hidden layer is selects + adds, no multiplies in the ternary path).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryEnergy {
    /// 2×2 quadratic form P, row-major [p00, p01, p10, p11].
    pub p: [f64; 4],
    /// input scale s applied before the ternary projection.
    pub scale: f64,
    /// ternary weights T, row-major 2·h (h hidden units), each in {−1,0,1}.
    pub t: Vec<i8>,
    /// hidden biases b₁ (length h).
    pub b1: Vec<f64>,
    /// output weights w₂ (length h).
    pub w2: Vec<f64>,
    /// energy offset v₀ (so V(0)=0 at the certified equilibrium).
    pub v0: f64,
}

/// The certificate a pack carries — the manifest facet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateSpec {
    /// Verifier id. v1: `"lyapunov-ternary-taylor-crown"`.
    pub kind: String,
    /// Closed-loop system id whose (baked) dynamics the certificate is proven against.
    /// v1: `"saturated-hybrid-wall-contact"` (free/contact mode switch + actuator saturation).
    pub system: String,
    /// Certified region as an annulus [r_inner, r_outer] in error-state radius.
    pub region: [f64; 2],
    /// Decrease margin coefficient α in ΔV ≤ −α‖e‖² (train-stricter-than-verify).
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    /// The certified energy weights.
    pub energy: TernaryEnergy,
}
fn default_alpha() -> f64 {
    5e-4
}

/// Outcome of an on-device re-verification.
#[derive(Debug, Clone)]
pub struct CertReport {
    pub certified: bool,
    /// number of boxes proven (sound ΔV<0) across the whole annulus.
    pub boxes: u64,
    /// deepest refinement level reached.
    pub depth: u32,
    /// worst (largest) ΔV+α‖e‖² upper bound seen at the final level.
    pub worst_bound: f64,
    /// if refuted, the center+radius of a box that could not be certified.
    pub bad_box: Option<[f64; 3]>,
}

#[derive(Debug, thiserror::Error)]
pub enum CertError {
    #[error("unknown certificate kind {0:?}")]
    Kind(String),
    #[error("unknown certified system {0:?}")]
    System(String),
    #[error("malformed energy: {0}")]
    Energy(String),
    #[error("certificate did NOT re-verify: box [{cx:.3},{cy:.3}]±{r:.3} has ΔV+α‖e‖² ≥ 0")]
    Refuted { cx: f64, cy: f64, r: f64 },
}

// ---- baked closed-loop dynamics, selected by `system` id ----
struct Sys {
    xw: f64,
    gb: f64,
    ks: f64,
    cc: f64,
    bd: f64,
    dt: f64,
    um: f64,
    kx: f64,
    kv: f64,
}
fn system(id: &str) -> Option<Sys> {
    match id {
        "saturated-hybrid-wall-contact" => Some(Sys {
            xw: 1.0, gb: 0.6, ks: 60.0, cc: 10.0, bd: 0.5, dt: 0.02, um: 4.0, kx: 8.0, kv: 3.0,
        }),
        _ => None,
    }
}

#[inline]
fn tf(e: &TernaryEnergy, j: usize, k: usize) -> f64 {
    e.t[2 * j + k] as f64
}
/// V(e)
fn vfn(e: &TernaryEnergy, e1: f64, e2: f64) -> f64 {
    let p = &e.p;
    let mut v = e1 * (p[0] * e1 + p[1] * e2) + e2 * (p[2] * e1 + p[3] * e2);
    for j in 0..e.b1.len() {
        v += e.w2[j] * (e.scale * (tf(e, j, 0) * e1 + tf(e, j, 1) * e2) + e.b1[j]).tanh();
    }
    v - e.v0
}
/// ∇V(e)
fn grad_v(e: &TernaryEnergy, e1: f64, e2: f64) -> (f64, f64) {
    let p = &e.p;
    let (mut g1, mut g2) = (2.0 * (p[0] * e1 + p[1] * e2), 2.0 * (p[2] * e1 + p[3] * e2));
    for j in 0..e.b1.len() {
        let th = (e.scale * (tf(e, j, 0) * e1 + tf(e, j, 1) * e2) + e.b1[j]).tanh();
        let d = e.w2[j] * (1.0 - th * th) * e.scale;
        g1 += d * tf(e, j, 0);
        g2 += d * tf(e, j, 1);
    }
    (g1, g2)
}
/// one closed-loop step; mode 0=free/1=contact; clamp -1=u −UM / 0=linear / +1=u +UM
fn step(s: &Sys, e1: f64, e2: f64, mode: usize, clamp: i32) -> (f64, f64) {
    let u = if clamp == -1 { -s.um } else if clamp == 1 { s.um } else { -s.gb - s.kx * e1 - s.kv * e2 };
    let mut a = s.gb + u - s.bd * e2;
    if mode == 1 { a -= s.ks * e1 + s.cc * e2; }
    let v2 = e2 + s.dt * a;
    (e1 + s.dt * v2, v2)
}
/// closed-loop Jacobian (constant per case — dynamics are affine)
fn jf(s: &Sys, mode: usize, clamp: i32) -> [[f64; 2]; 2] {
    let (mut da1, mut da2) = (0.0, -s.bd);
    if clamp == 0 { da1 += -s.kx; da2 += -s.kv; }
    if mode == 1 { da1 += -s.ks; da2 += -s.cc; }
    let (dv1, dv2) = (s.dt * da1, 1.0 + s.dt * da2);
    [[1.0 + s.dt * dv1, s.dt * dv2], [dv1, dv2]]
}
/// tight per-box bound on |tanh″(z)| over z∈[lo,hi]; peak 0.7698 at |z|=0.6585
fn d2max(lo: f64, hi: f64) -> f64 {
    let (tl, th) = (lo.tanh(), hi.tanh());
    let m = (2.0 * tl.abs() * (1.0 - tl * tl)).max(2.0 * th.abs() * (1.0 - th * th));
    if (lo <= 0.6585 && hi >= 0.6585) || (lo <= -0.6585 && hi >= -0.6585) { 0.7698 } else { m }
}
/// 2nd-order Taylor + CROWN upper bound on ΔV+α‖e‖² over a box, one hybrid case
#[allow(clippy::too_many_arguments)]
fn bound(s: &Sys, e: &TernaryEnergy, alpha: f64, c1: f64, c2: f64, r1: f64, r2: f64, mode: usize, clamp: i32) -> f64 {
    let (fx, fy) = step(s, c1, c2, mode, clamp);
    let dvc = vfn(e, fx, fy) - vfn(e, c1, c2);
    let (gfx, gfy) = grad_v(e, fx, fy);
    let (gsx, gsy) = grad_v(e, c1, c2);
    let j = jf(s, mode, clamp);
    let gd1 = j[0][0] * gfx + j[1][0] * gfy - gsx;
    let gd2 = j[0][1] * gfx + j[1][1] * gfy - gsy;
    let p2 = [[2.0 * e.p[0], 2.0 * e.p[1]], [2.0 * e.p[2], 2.0 * e.p[3]]];
    let mut pj = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { pj[i][k] = p2[i][0] * j[0][k] + p2[i][1] * j[1][k]; } }
    let mut m = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { m[i][k] = j[0][i] * pj[0][k] + j[1][i] * pj[1][k] - p2[i][k]; } }
    let mut hs = [[0.0; 2]; 2]; let mut hfm = [[0.0; 2]; 2];
    let aj = [[j[0][0].abs(), j[0][1].abs()], [j[1][0].abs(), j[1][1].abs()]];
    let (fr1, fr2) = (aj[0][0] * r1 + aj[0][1] * r2, aj[1][0] * r1 + aj[1][1] * r2);
    for jx in 0..e.b1.len() {
        let (a0, a1) = (tf(e, jx, 0).abs(), tf(e, jx, 1).abs());
        let zc = e.scale * (tf(e, jx, 0) * c1 + tf(e, jx, 1) * c2) + e.b1[jx]; let zr = e.scale * (a0 * r1 + a1 * r2);
        let cs = e.w2[jx].abs() * d2max(zc - zr, zc + zr) * e.scale * e.scale;
        hs[0][0] += cs * a0 * a0; hs[0][1] += cs * a0 * a1; hs[1][0] += cs * a1 * a0; hs[1][1] += cs * a1 * a1;
        let zcf = e.scale * (tf(e, jx, 0) * fx + tf(e, jx, 1) * fy) + e.b1[jx]; let zrf = e.scale * (a0 * fr1 + a1 * fr2);
        let cf = e.w2[jx].abs() * d2max(zcf - zrf, zcf + zrf) * e.scale * e.scale;
        hfm[0][0] += cf * a0 * a0; hfm[0][1] += cf * a0 * a1; hfm[1][0] += cf * a1 * a0; hfm[1][1] += cf * a1 * a1;
    }
    let mut hfj = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { hfj[i][k] = hfm[i][0] * aj[0][k] + hfm[i][1] * aj[1][k]; } }
    let mut habs = [[0.0; 2]; 2];
    for i in 0..2 { for k in 0..2 { habs[i][k] = m[i][k].abs() + hs[i][k] + (aj[0][i] * hfj[0][k] + aj[1][i] * hfj[1][k]); } }
    let ss_hi = (c1.abs() + r1).powi(2) + (c2.abs() + r2).powi(2);
    let rem = 0.5 * (habs[0][0] * r1 * r1 + habs[0][1] * r1 * r2 + habs[1][0] * r2 * r1 + habs[1][1] * r2 * r2);
    dvc + (gd1.abs() * r1 + gd2.abs() * r2) + rem + alpha * ss_hi
}
fn case_active(s: &Sys, c1: f64, c2: f64, r1: f64, r2: f64, mode: usize, clamp: i32) -> bool {
    let (x_lo, x_hi) = (c1 + s.xw - r1, c1 + s.xw + r1);
    let mode_ok = if mode == 0 { x_lo < s.xw } else { x_hi >= s.xw };
    let ur_c = -s.gb - s.kx * c1 - s.kv * c2; let ur_r = s.kx * r1 + s.kv * r2;
    let clamp_ok = match clamp {
        -1 => ur_c - ur_r <= -s.um,
        1 => ur_c + ur_r >= s.um,
        _ => ur_c - ur_r <= s.um && ur_c + ur_r >= -s.um,
    };
    mode_ok && clamp_ok
}
fn in_region(r_in: f64, r_out: f64, c1: f64, c2: f64, r1: f64, r2: f64) -> bool {
    let lo = (c1.abs() - r1).max(0.0).powi(2) + (c2.abs() - r2).max(0.0).powi(2);
    let hi = (c1.abs() + r1).powi(2) + (c2.abs() + r2).powi(2);
    hi >= r_in * r_in && lo <= r_out * r_out
}

/// Re-prove the certificate ON THIS DEVICE. Pure f64 + tanh, no solver, wasm-clean.
/// Ok(report) with `report.certified == true` ⇒ the pack's energy still carries a valid
/// Lyapunov certificate over the whole declared annulus (all hybrid cases). Err(Refuted{..})
/// names a box that fails — a drifted/tampered energy is rejected exactly as a bad eval vector is.
pub fn reverify(spec: &CertificateSpec) -> Result<CertReport, CertError> {
    if spec.kind != "lyapunov-ternary-taylor-crown" {
        return Err(CertError::Kind(spec.kind.clone()));
    }
    let s = system(&spec.system).ok_or_else(|| CertError::System(spec.system.clone()))?;
    let e = &spec.energy;
    let h = e.b1.len();
    if h == 0 || e.w2.len() != h || e.t.len() != 2 * h {
        return Err(CertError::Energy(format!(
            "inconsistent lengths: b1={}, w2={}, t={} (need 2·b1)",
            e.b1.len(), e.w2.len(), e.t.len()
        )));
    }
    if e.t.iter().any(|&v| v < -1 || v > 1) {
        return Err(CertError::Energy("T has an entry outside {−1,0,1}".into()));
    }
    let (r_in, r_out) = (spec.region[0], spec.region[1]);
    if !(r_out > r_in && r_in >= 0.0) {
        return Err(CertError::Energy(format!("bad region [{r_in}, {r_out}]")));
    }

    // adaptive box refinement over the annulus
    let h0 = 0.06f64;
    let mut boxes: Vec<[f64; 4]> = Vec::new();
    let n = (2.0 * r_out / h0).ceil() as i64;
    for i in 0..n {
        for k in 0..n {
            let c1 = -r_out + (i as f64 + 0.5) * h0;
            let c2 = -r_out + (k as f64 + 0.5) * h0;
            if in_region(r_in, r_out, c1, c2, h0 / 2.0, h0 / 2.0) {
                boxes.push([c1, c2, h0 / 2.0, h0 / 2.0]);
            }
        }
    }
    let cap = 4_000_000usize;
    let mut certified = 0u64;
    let mut depth = 0u32;
    loop {
        let mut fails: Vec<[f64; 4]> = Vec::new();
        let mut worst_b = f64::NEG_INFINITY;
        for b in &boxes {
            let (c1, c2, r1, r2) = (b[0], b[1], b[2], b[3]);
            let mut ok = true;
            for mode in 0..2 {
                for clamp in -1..=1 {
                    if case_active(&s, c1, c2, r1, r2, mode, clamp) {
                        let bd = bound(&s, e, spec.alpha, c1, c2, r1, r2, mode, clamp);
                        if bd > worst_b { worst_b = bd; }
                        if bd >= 0.0 { ok = false; }
                    }
                }
            }
            if ok { certified += 1; } else { fails.push(*b); }
        }
        if fails.is_empty() {
            return Ok(CertReport { certified: true, boxes: certified, depth, worst_bound: worst_b, bad_box: None });
        }
        if fails.len() > cap || depth >= 20 {
            let b = fails[0];
            return Err(CertError::Refuted { cx: b[0], cy: b[1], r: b[2] });
        }
        let mut next: Vec<[f64; 4]> = Vec::with_capacity(fails.len() * 4);
        for b in &fails {
            let (nr1, nr2) = (b[2] / 2.0, b[3] / 2.0);
            for &sx in &[-1.0, 1.0] {
                for &sy in &[-1.0, 1.0] {
                    let (c1, c2) = (b[0] + sx * nr1, b[1] + sy * nr2);
                    if in_region(r_in, r_out, c1, c2, nr1, nr2) {
                        next.push([c1, c2, nr1, nr2]);
                    }
                }
            }
        }
        boxes = next;
        depth += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the certified saturated-hybrid ternary energy (certified_sat_taylor_R1.2.npz), embedded.
    fn certified_spec() -> CertificateSpec {
        CertificateSpec {
            kind: "lyapunov-ternary-taylor-crown".into(),
            system: "saturated-hybrid-wall-contact".into(),
            region: [0.15, 1.2],
            alpha: 5e-4,
            energy: TernaryEnergy {
                p: [31.988, 2.543, 2.543, 1.4169999999999998],
                scale: 1.4740514336487223,
                t: vec![-1, 0, -1, 0, 0, 0, -1, -1, -1, 0, 1, 1, 1, 0, 0, 0],
                b1: vec![-1.7253050443315214, -1.6895583892614585, -1.572812794286765, -2.9962607818256326,
                         -1.362285297766529, 2.9948712315814445, 1.6641829682841895, 0.15496976355688338],
                w2: vec![-1.8092778484049137, -0.6474111753241604, -0.0407591631059273, 1.2424339130389697,
                         -0.3175122724570142, -1.1023667519975417, 0.6676446220380198, -5.977279221172131e-12],
                v0: 0.9069008854718814,
            },
        }
    }

    #[test]
    fn energy_matches_reference() {
        // cross-check V against the certified reference values (bit-faithful f64 port)
        let e = certified_spec().energy;
        let refs = [([0.5, -0.5], 7.262593626850), ([-0.3, 0.8], 2.244816459922), ([0.9, 0.4], 28.178306136649)];
        for (pt, r) in refs {
            assert!((vfn(&e, pt[0], pt[1]) - r).abs() < 1e-8, "V({pt:?}) drifted from certified reference");
        }
    }

    #[test]
    fn certified_energy_reverifies() {
        let rep = reverify(&certified_spec()).expect("the certified energy must re-verify");
        assert!(rep.certified);
        assert!(rep.worst_bound < 0.0, "worst bound must be negative, got {}", rep.worst_bound);
        assert!(rep.boxes > 1000);
    }

    #[test]
    fn tampered_energy_is_refuted() {
        // perturb one output weight — the deployed energy no longer certifies
        let mut spec = certified_spec();
        spec.energy.w2[0] += 0.5;
        match reverify(&spec) {
            Err(CertError::Refuted { .. }) => {}
            other => panic!("expected Refuted, got {other:?}"),
        }
    }

    #[test]
    fn bare_quadratic_refuted_at_r12() {
        // drop the learned head: the quadratic alone is refuted past R≈1.0 (the §5 law)
        let mut spec = certified_spec();
        for w in spec.energy.w2.iter_mut() { *w = 0.0; }
        spec.energy.v0 = 0.0;
        assert!(matches!(reverify(&spec), Err(CertError::Refuted { .. })));
    }

    #[test]
    fn unknown_system_rejected() {
        let mut spec = certified_spec();
        spec.system = "some-other-robot".into();
        assert!(matches!(reverify(&spec), Err(CertError::System(_))));
    }
}
