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
//! weights from the manifest, over a small **registry of certified systems** (see [`Sys`]). The
//! energy V(e)=eᵀPe + Σⱼ w₂ⱼ·tanh(s·(Tⱼ·e)+b₁ⱼ) − v₀ spans both regimes of the law: on a smooth
//! (convex-ROA) system the ternary head is empty and the quadratic alone certifies; on a
//! non-convex/hybrid system the head is what earns the extra region. Synthesis lineage: Chang/Gao
//! (Neural Lyapunov Control, dReal); our legs are the ternary weights, the quadratic-anchored init,
//! and train-stricter-than-verify margins. SOS/dReal stay a build-time/fleet gate — they need
//! solvers; this Taylor+CROWN pass is the on-device gate.

use serde::{Deserialize, Serialize};

/// A learned ternary Lyapunov energy: V(e) = eᵀPe + Σⱼ w₂ⱼ·tanh(s·(Tⱼ·e)+b₁ⱼ) − v₀,
/// with T ∈ {−1,0,+1} (the hidden layer is selects + adds, no multiplies in the ternary path).
/// The head may be empty (h = 0) — then V is a pure quadratic (the convex-ROA regime of the law).
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
    /// v1: `"saturated-hybrid-wall-contact"` (free/contact mode switch + actuator saturation),
    /// `"linear-double-integrator"` (smooth LQR double integrator).
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

// ---- closed-loop dynamics: one variant per certified system, dispatched by `system` id ----
#[derive(Clone, Copy)]
enum Sys {
    /// wall-contact free/contact mode switch + gravity bias + actuator saturation (6 cases, affine).
    Hybrid { xw: f64, gb: f64, ks: f64, cc: f64, bd: f64, dt: f64, um: f64, kx: f64, kv: f64 },
    /// pure double integrator under an LQR-style law: smooth, one case, no saturation, no wall (affine).
    LinearDi { dt: f64, kx: f64, kv: f64 },
    /// double integrator with ACTUATOR SATURATION (3 clamp cases, affine): the plain quadratic is
    /// refuted past R≈0.8; a trained ternary head certifies further (the learned-beats-quadratic case).
    SaturatedDi { dt: f64, kx: f64, kv: f64, um: f64 },
    /// reversed Van der Pol: ẋ=[−x₂, x₁+(x₁²−1)x₂] — a genuinely NONLINEAR vector field with a
    /// non-convex region of attraction (the flagship "learned > quadratic" case). One case; the
    /// Jacobian varies with state, so the Taylor bound adds a state-box Jacobian range and the
    /// dynamics' own Hessian term.
    VanDerPol { dt: f64 },
}
fn system(id: &str) -> Option<Sys> {
    match id {
        "saturated-hybrid-wall-contact" => Some(Sys::Hybrid {
            xw: 1.0, gb: 0.6, ks: 60.0, cc: 10.0, bd: 0.5, dt: 0.02, um: 4.0, kx: 8.0, kv: 3.0,
        }),
        "linear-double-integrator" => Some(Sys::LinearDi { dt: 0.02, kx: 8.0, kv: 3.0 }),
        "saturated-double-integrator" => Some(Sys::SaturatedDi { dt: 0.02, kx: 8.0, kv: 3.0, um: 4.0 }),
        "reversed-van-der-pol" => Some(Sys::VanDerPol { dt: 0.02 }),
        _ => None,
    }
}
impl Sys {
    /// the (mode, clamp) cases that partition this system's dynamics.
    fn cases(&self) -> &'static [(usize, i32)] {
        match self {
            Sys::Hybrid { .. } => &[(0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)],
            Sys::SaturatedDi { .. } => &[(0, -1), (0, 0), (0, 1)],
            Sys::LinearDi { .. } | Sys::VanDerPol { .. } => &[(0, 0)],
        }
    }
    /// affine systems have a constant Jacobian; the Van der Pol's varies with state.
    fn is_affine(&self) -> bool {
        !matches!(self, Sys::VanDerPol { .. })
    }
    fn vdp_dt(&self) -> f64 {
        match *self { Sys::VanDerPol { dt } => dt, _ => 0.0 }
    }
    /// one closed-loop step; hybrid: mode 0=free/1=contact, clamp -1=u −UM / 0=linear / +1=u +UM.
    fn step(&self, e1: f64, e2: f64, mode: usize, clamp: i32) -> (f64, f64) {
        match *self {
            Sys::Hybrid { gb, ks, cc, bd, dt, um, kx, kv, .. } => {
                let u = if clamp == -1 { -um } else if clamp == 1 { um } else { -gb - kx * e1 - kv * e2 };
                let mut a = gb + u - bd * e2;
                if mode == 1 { a -= ks * e1 + cc * e2; }
                let v2 = e2 + dt * a;
                (e1 + dt * v2, v2)
            }
            Sys::LinearDi { dt, kx, kv } => {
                let a = -kx * e1 - kv * e2; // no bias, no saturation, no contact
                let v2 = e2 + dt * a;
                (e1 + dt * v2, v2)
            }
            Sys::SaturatedDi { dt, kx, kv, um } => {
                let u = if clamp == -1 { -um } else if clamp == 1 { um } else { -kx * e1 - kv * e2 };
                let v2 = e2 + dt * u;
                (e1 + dt * v2, v2)
            }
            Sys::VanDerPol { dt } => (e1 - dt * e2, e2 + dt * (e1 + (e1 * e1 - 1.0) * e2)),
        }
    }
    /// closed-loop Jacobian AT the state (affine variants ignore e1,e2).
    fn jf(&self, e1: f64, e2: f64, mode: usize, clamp: i32) -> [[f64; 2]; 2] {
        match *self {
            Sys::Hybrid { ks, cc, bd, dt, kx, kv, .. } => {
                let (mut da1, mut da2) = (0.0, -bd);
                if clamp == 0 { da1 += -kx; da2 += -kv; }
                if mode == 1 { da1 += -ks; da2 += -cc; }
                let (dv1, dv2) = (dt * da1, 1.0 + dt * da2);
                [[1.0 + dt * dv1, dt * dv2], [dv1, dv2]]
            }
            Sys::LinearDi { dt, kx, kv } => {
                let (da1, da2) = (-kx, -kv);
                let (dv1, dv2) = (dt * da1, 1.0 + dt * da2);
                [[1.0 + dt * dv1, dt * dv2], [dv1, dv2]]
            }
            Sys::SaturatedDi { dt, kx, kv, .. } => {
                let (da1, da2) = if clamp == 0 { (-kx, -kv) } else { (0.0, 0.0) };
                let (dv1, dv2) = (dt * da1, 1.0 + dt * da2);
                [[1.0 + dt * dv1, dt * dv2], [dv1, dv2]]
            }
            Sys::VanDerPol { dt } => {
                [[1.0, -dt], [dt * (1.0 + 2.0 * e1 * e2), 1.0 + dt * (e1 * e1 - 1.0)]]
            }
        }
    }
    /// entrywise max |J| over the box (affine: |constant J|; Van der Pol: bounded state range).
    fn j_absbox(&self, c1: f64, c2: f64, r1: f64, r2: f64, mode: usize, clamp: i32) -> [[f64; 2]; 2] {
        match *self {
            Sys::VanDerPol { dt } => {
                let (a1l, a1h, a2l, a2h) = (c1 - r1, c1 + r1, c2 - r2, c2 + r2);
                let prods = [2.0 * a1l * a2l, 2.0 * a1l * a2h, 2.0 * a1h * a2l, 2.0 * a1h * a2h];
                let (mut pmin, mut pmax) = (f64::INFINITY, f64::NEG_INFINITY);
                for &p in &prods { pmin = pmin.min(p); pmax = pmax.max(p); }
                let j10 = dt * (1.0 + pmin).abs().max((1.0 + pmax).abs());
                let e1sq_hi = (a1l * a1l).max(a1h * a1h);
                let e1sq_lo = if a1l * a1h <= 0.0 { 0.0 } else { a1l.abs().min(a1h.abs()).powi(2) };
                let j11 = (1.0 + dt * (e1sq_hi - 1.0)).abs().max((1.0 + dt * (e1sq_lo - 1.0)).abs());
                [[1.0, dt], [j10, j11]]
            }
            _ => {
                let j = self.jf(c1, c2, mode, clamp);
                [[j[0][0].abs(), j[0][1].abs()], [j[1][0].abs(), j[1][1].abs()]]
            }
        }
    }
    /// is this case reachable over the given box? (mode/saturation partition; single-case = always).
    fn case_active(&self, c1: f64, c2: f64, r1: f64, r2: f64, mode: usize, clamp: i32) -> bool {
        match *self {
            Sys::Hybrid { xw, gb, um, kx, kv, .. } => {
                let (x_lo, x_hi) = (c1 + xw - r1, c1 + xw + r1);
                let mode_ok = if mode == 0 { x_lo < xw } else { x_hi >= xw };
                let ur_c = -gb - kx * c1 - kv * c2;
                let ur_r = kx * r1 + kv * r2;
                let clamp_ok = match clamp {
                    -1 => ur_c - ur_r <= -um,
                    1 => ur_c + ur_r >= um,
                    _ => ur_c - ur_r <= um && ur_c + ur_r >= -um,
                };
                mode_ok && clamp_ok
            }
            Sys::SaturatedDi { um, kx, kv, .. } => {
                let ur_c = -kx * c1 - kv * c2;
                let ur_r = kx * r1 + kv * r2;
                match clamp {
                    -1 => ur_c - ur_r <= -um,
                    1 => ur_c + ur_r >= um,
                    _ => ur_c - ur_r <= um && ur_c + ur_r >= -um,
                }
            }
            Sys::LinearDi { .. } | Sys::VanDerPol { .. } => true,
        }
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
/// tight per-box bound on |tanh″(z)| over z∈[lo,hi]; peak 0.7698 at |z|=0.6585
fn d2max(lo: f64, hi: f64) -> f64 {
    let (tl, th) = (lo.tanh(), hi.tanh());
    let m = (2.0 * tl.abs() * (1.0 - tl * tl)).max(2.0 * th.abs() * (1.0 - th * th));
    if (lo <= 0.6585 && hi >= 0.6585) || (lo <= -0.6585 && hi >= -0.6585) { 0.7698 } else { m }
}
/// 2nd-order Taylor + CROWN upper bound on ΔV+α‖e‖² over a box, one case.
///
/// The center gradient (JᵀgV(f) − gV(c), exact at the box center) and the ternary-head CROWN
/// bounds (`hs` at the box, `hfm` at f(box)) are shared. The QUADRATIC part of the Hessian differs:
/// for an affine system the constant Jacobian lets JᵀPJ−P cancel exactly (tight); for the nonlinear
/// Van der Pol the Jacobian ranges over the box, so we use the sound conservative bound
/// |J|ᵀ|H_V(f)||J| + |H_V(c)| and add the dynamics' own Hessian term (∇V(f))₂·H_{f₂}.
#[allow(clippy::too_many_arguments)]
fn bound(s: &Sys, e: &TernaryEnergy, alpha: f64, c1: f64, c2: f64, r1: f64, r2: f64, mode: usize, clamp: i32) -> f64 {
    let (fx, fy) = s.step(c1, c2, mode, clamp);
    let dvc = vfn(e, fx, fy) - vfn(e, c1, c2);
    let (gfx, gfy) = grad_v(e, fx, fy);
    let (gsx, gsy) = grad_v(e, c1, c2);
    let j = s.jf(c1, c2, mode, clamp); // Jacobian at the box center
    let gd1 = j[0][0] * gfx + j[1][0] * gfy - gsx;
    let gd2 = j[0][1] * gfx + j[1][1] * gfy - gsy;
    let p2 = [[2.0 * e.p[0], 2.0 * e.p[1]], [2.0 * e.p[2], 2.0 * e.p[3]]];
    // |J| over the box (= |center J| for affine systems), and the induced f-range radius.
    let aj = s.j_absbox(c1, c2, r1, r2, mode, clamp);
    let (fr1, fr2) = (aj[0][0] * r1 + aj[0][1] * r2, aj[1][0] * r1 + aj[1][1] * r2);
    // ternary-head Hessian bounds (CROWN |tanh″|) at the box (hs) and at f(box) (hfm).
    let mut hs = [[0.0; 2]; 2]; let mut hfm = [[0.0; 2]; 2];
    for jx in 0..e.b1.len() {
        let (a0, a1) = (tf(e, jx, 0).abs(), tf(e, jx, 1).abs());
        let zc = e.scale * (tf(e, jx, 0) * c1 + tf(e, jx, 1) * c2) + e.b1[jx]; let zr = e.scale * (a0 * r1 + a1 * r2);
        let cs = e.w2[jx].abs() * d2max(zc - zr, zc + zr) * e.scale * e.scale;
        hs[0][0] += cs * a0 * a0; hs[0][1] += cs * a0 * a1; hs[1][0] += cs * a1 * a0; hs[1][1] += cs * a1 * a1;
        let zcf = e.scale * (tf(e, jx, 0) * fx + tf(e, jx, 1) * fy) + e.b1[jx]; let zrf = e.scale * (a0 * fr1 + a1 * fr2);
        let cf = e.w2[jx].abs() * d2max(zcf - zrf, zcf + zrf) * e.scale * e.scale;
        hfm[0][0] += cf * a0 * a0; hfm[0][1] += cf * a0 * a1; hfm[1][0] += cf * a1 * a0; hfm[1][1] += cf * a1 * a1;
    }
    let mut habs = [[0.0; 2]; 2];
    if s.is_affine() {
        // exact-cancellation quadratic Hessian: m = JᵀP₂J − P₂ (J constant), + head via |J|.
        let mut pj = [[0.0; 2]; 2];
        for i in 0..2 { for k in 0..2 { pj[i][k] = p2[i][0] * j[0][k] + p2[i][1] * j[1][k]; } }
        let mut m = [[0.0; 2]; 2];
        for i in 0..2 { for k in 0..2 { m[i][k] = j[0][i] * pj[0][k] + j[1][i] * pj[1][k] - p2[i][k]; } }
        let mut hfj = [[0.0; 2]; 2];
        for i in 0..2 { for k in 0..2 { hfj[i][k] = hfm[i][0] * aj[0][k] + hfm[i][1] * aj[1][k]; } }
        for i in 0..2 { for k in 0..2 { habs[i][k] = m[i][k].abs() + hs[i][k] + (aj[0][i] * hfj[0][k] + aj[1][i] * hfj[1][k]); } }
    } else {
        // nonlinear: conservative |J|ᵀ(|2P|+hfm)|J| + (|2P|+hs), plus the dynamics-Hessian term.
        let p2a = [[p2[0][0].abs(), p2[0][1].abs()], [p2[1][0].abs(), p2[1][1].abs()]];
        let mut hvf = [[0.0; 2]; 2]; let mut hvc = [[0.0; 2]; 2];
        for i in 0..2 { for k in 0..2 { hvf[i][k] = p2a[i][k] + hfm[i][k]; hvc[i][k] = p2a[i][k] + hs[i][k]; } }
        let mut hj = [[0.0; 2]; 2];
        for i in 0..2 { for k in 0..2 { hj[i][k] = hvf[i][0] * aj[0][k] + hvf[i][1] * aj[1][k]; } }
        for i in 0..2 { for k in 0..2 { habs[i][k] = (aj[0][i] * hj[0][k] + aj[1][i] * hj[1][k]) + hvc[i][k]; } }
        // Σ_k (∇V(f))_k H_{f_k}: only f₂ curves. H_{f₂}=2·dt·[[e₂,e₁],[e₁,0]]; bound |∇V(f)₂| over box.
        let gfy_bound = gfy.abs() + hvf[1][0] * fr1 + hvf[1][1] * fr2;
        let (e1m, e2m) = (c1.abs() + r1, c2.abs() + r2);
        let coef = 2.0 * s.vdp_dt() * gfy_bound;
        habs[0][0] += coef * e2m; habs[0][1] += coef * e1m; habs[1][0] += coef * e1m;
    }
    let ss_hi = (c1.abs() + r1).powi(2) + (c2.abs() + r2).powi(2);
    let rem = 0.5 * (habs[0][0] * r1 * r1 + habs[0][1] * r1 * r2 + habs[1][0] * r2 * r1 + habs[1][1] * r2 * r2);
    dvc + (gd1.abs() * r1 + gd2.abs() * r2) + rem + alpha * ss_hi
}
fn in_region(r_in: f64, r_out: f64, c1: f64, c2: f64, r1: f64, r2: f64) -> bool {
    let lo = (c1.abs() - r1).max(0.0).powi(2) + (c2.abs() - r2).max(0.0).powi(2);
    let hi = (c1.abs() + r1).powi(2) + (c2.abs() + r2).powi(2);
    hi >= r_in * r_in && lo <= r_out * r_out
}

/// Re-prove the certificate ON THIS DEVICE. Pure f64 + tanh, no solver, wasm-clean.
/// Ok(report) with `report.certified == true` ⇒ the pack's energy still carries a valid
/// Lyapunov certificate over the whole declared annulus (all of the system's cases). Err(Refuted{..})
/// names a box that fails — a drifted/tampered energy is rejected exactly as a bad eval vector is.
pub fn reverify(spec: &CertificateSpec) -> Result<CertReport, CertError> {
    if spec.kind != "lyapunov-ternary-taylor-crown" {
        return Err(CertError::Kind(spec.kind.clone()));
    }
    let s = system(&spec.system).ok_or_else(|| CertError::System(spec.system.clone()))?;
    let e = &spec.energy;
    let h = e.b1.len();
    if e.w2.len() != h || e.t.len() != 2 * h {
        return Err(CertError::Energy(format!(
            "inconsistent lengths: b1={}, w2={}, t={} (need w2==b1 and t==2·b1)",
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
            for &(mode, clamp) in s.cases() {
                if s.case_active(c1, c2, r1, r2, mode, clamp) {
                    let bd = bound(&s, e, spec.alpha, c1, c2, r1, r2, mode, clamp);
                    if bd > worst_b { worst_b = bd; }
                    if bd >= 0.0 { ok = false; }
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

    /// the smooth linear double integrator with its exact discrete-Lyapunov quadratic (no head).
    fn linear_di_spec() -> CertificateSpec {
        CertificateSpec {
            kind: "lyapunov-ternary-taylor-crown".into(),
            system: "linear-double-integrator".into(),
            region: [0.15, 2.0],
            alpha: 5e-4,
            energy: TernaryEnergy {
                // P = dlyap(J^T, I) for the LQR closed loop (J^T P J − P = −I exactly)
                p: [84.186700371440466, 2.2125206355757552, 2.2125206355757534, 9.5185075319851435],
                scale: 1.0,
                t: vec![],
                b1: vec![],
                w2: vec![],
                v0: 0.0,
            },
        }
    }

    #[test]
    fn energy_matches_reference() {
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
        // corrupt an output weight decisively (as a flipped byte would) — the deployed energy
        // no longer certifies. (A *small* perturbation still certifies: the certificate carries
        // a train-stricter-than-verify margin, which is a feature, not a miss.)
        let mut spec = certified_spec();
        spec.energy.w2[0] = 5.0;
        match reverify(&spec) {
            Err(CertError::Refuted { .. }) => {}
            other => panic!("expected Refuted, got {other:?}"),
        }
    }

    #[test]
    fn bare_quadratic_refuted_at_r12() {
        // drop the learned head on the HYBRID system: the quadratic alone is refuted past R≈1.0.
        let mut spec = certified_spec();
        for w in spec.energy.w2.iter_mut() { *w = 0.0; }
        spec.energy.v0 = 0.0;
        assert!(matches!(reverify(&spec), Err(CertError::Refuted { .. })));
    }

    #[test]
    fn linear_double_integrator_reverifies_with_pure_quadratic() {
        // the second certified system: smooth, one case, EMPTY ternary head — the quadratic
        // alone certifies (the convex-ROA regime of the law). Proves the facet is a registry,
        // not a single hardcoded system.
        let rep = reverify(&linear_di_spec()).expect("the linear double integrator must re-verify");
        assert!(rep.certified);
        assert!(rep.worst_bound < 0.0);
    }

    #[test]
    fn unknown_system_rejected() {
        let mut spec = certified_spec();
        spec.system = "some-other-robot".into();
        assert!(matches!(reverify(&spec), Err(CertError::System(_))));
    }

    /// the reversed Van der Pol certificate (certified_R1.3.npz). Its dense W1 is exactly
    /// scale·T with T∈{−1,0,1} (scale = 3.4601597785949707), so it is a ternary energy.
    fn van_der_pol_spec() -> CertificateSpec {
        CertificateSpec {
            kind: "lyapunov-ternary-taylor-crown".into(),
            system: "reversed-van-der-pol".into(),
            region: [0.15, 1.3],
            alpha: 5e-4,
            energy: TernaryEnergy {
                p: [76.2755359693619, -25.51278060966837, -25.51278060966837, 51.27806096683688],
                scale: 3.4601597785949707,
                t: vec![-1, 0, 0, -1, 1, 0, 0, -1, 0, -1, -1, 0, 0, 1, 1, 0],
                b1: vec![-2.3375589847564697, -2.102060556411743, -2.708591938018799, -2.0922701358795166,
                         -2.0625295639038086, 1.6640418767929077, -2.6110639572143555, -2.7082889080047607],
                w2: vec![-7.458496570587158, 3.7638297080993652, -4.320842266082764, 3.884138584136963,
                         3.7923171520233154, 1.5532889366149902, 7.402223110198975, -4.4193596839904785],
                v0: -0.9857819872128699,
            },
        }
    }

    #[test]
    fn van_der_pol_energy_matches_dense_reference() {
        // scale·T must reconstruct the dense W1 energy bit-faithfully.
        let e = van_der_pol_spec().energy;
        let refs = [([0.5, -0.5], 48.254788539800), ([-0.3, 0.8], 59.325005796324), ([0.9, 0.4], 37.659142697147)];
        for (pt, r) in refs {
            assert!((vfn(&e, pt[0], pt[1]) - r).abs() < 1e-6, "V({pt:?})={} vs ref {r}", vfn(&e, pt[0], pt[1]));
        }
    }

    #[test]
    fn van_der_pol_nonlinear_reverifies() {
        // the flagship: a genuinely NONLINEAR, non-convex-ROA system certified on-device to R=1.3
        // (matching the dReal ground truth) by the state-box Jacobian + dynamics-Hessian extension.
        let rep = reverify(&van_der_pol_spec()).expect("the Van der Pol certificate must re-verify");
        assert!(rep.certified);
        assert!(rep.worst_bound < 0.0, "worst bound must be negative, got {}", rep.worst_bound);
        assert!(rep.boxes > 1000);
    }

    #[test]
    fn van_der_pol_bare_quadratic_refuted() {
        // head off: the quadratic alone has a real violation past R≈1.1, so R=1.3 must refute.
        let mut spec = van_der_pol_spec();
        for w in spec.energy.w2.iter_mut() { *w = 0.0; }
        spec.energy.v0 = 0.0;
        assert!(matches!(reverify(&spec), Err(CertError::Refuted { .. })));
    }

    /// the TRAINED ternary head for the saturated double integrator: it certifies a region the
    /// plain quadratic cannot (the quadratic is refuted past R≈0.8). Learned-beats-quadratic on a
    /// smooth-but-input-constrained system. Soundness Monte-Carlo-verified (true max −0.012).
    fn saturated_di_spec() -> CertificateSpec {
        CertificateSpec {
            kind: "lyapunov-ternary-taylor-crown".into(),
            system: "saturated-double-integrator".into(),
            region: [0.15, 1.0],
            alpha: 5e-4,
            energy: TernaryEnergy {
                p: [84.18670037144047, 2.212520635575755, 2.2125206355757534, 9.518507531985144],
                scale: 1.9497924248377483,
                t: vec![0, -1, 0, -1, 0, 1, 0, -1, 0, -1, 0, 1, 0, 1, 1, -1],
                b1: vec![-1.7654157876968384, -1.7829656600952148, -1.6165714263916016, 1.5079240798950195,
                         -1.8295671939849854, 2.0675222873687744, -1.7644506692886353, 1.109099268913269],
                w2: vec![1.0797884464263916, 1.277887225151062, 1.6327425241470337, -1.4929447174072266,
                         1.0733451843261719, -1.2776927947998047, 1.4379184246063232, 1.4926438331604004],
                v0: -7.501433930624267,
            },
        }
    }

    #[test]
    fn saturated_di_trained_head_reverifies() {
        // the learned head certifies R=1.0 (quadratic dies at ~0.8).
        let rep = reverify(&saturated_di_spec()).expect("the trained saturated-DI head must re-verify");
        assert!(rep.certified);
        assert!(rep.worst_bound < 0.0, "worst bound must be negative, got {}", rep.worst_bound);
    }

    #[test]
    fn saturated_di_bare_quadratic_refuted() {
        // head off: the plain quadratic is refuted at R=1.0 (real violation past ~0.8).
        let mut spec = saturated_di_spec();
        for w in spec.energy.w2.iter_mut() { *w = 0.0; }
        spec.energy.v0 = 0.0;
        assert!(matches!(reverify(&spec), Err(CertError::Refuted { .. })));
    }
}
