# certificate-example — a pack that carries a formal Lyapunov certificate

Ferralloy's eval vectors prove a pack *reproduces* signed outputs bit-for-bit across fabrics
(verified **behavior**). A certificate proves the axis beyond that: that the deployed control
energy still carries a valid formal **Lyapunov** guarantee — it drives the body into its basin
and keeps it there — re-proven **on the device** before the pack is trusted, with no SDP/SMT
solver and nothing from libm beyond `tanh`. See `crates/ferralloy-pack/src/certificate.rs`.

`cert.json` is the SMT/Taylor+CROWN-certified ternary energy from the Charlot Lab certificate
program (`certified_sat_taylor_R1.2.npz`): a learned `V(e)=eᵀPe + Σⱼ w₂ⱼ·tanh(s·(Tⱼ·e)+b₁ⱼ) − v₀`
with T ∈ {−1,0,+1}, certified as a Lyapunov function for the saturated-hybrid wall-contact system
over the annulus r ∈ [0.15, 1.2] — across both the free/contact mode switch and actuator saturation.

## Try it

```sh
ferralloy keygen                                    # once
# Build a pack that carries the certificate (the build-time gate re-proves it):
ferralloy build payloads/certificate-example/policy \
    --name reach-certified --entry policy.wasm \
    --certificate payloads/certificate-example/cert.json -o reach.fpack
# → certificate: CERTIFIED (1486 boxes, worst bound -0.00000)

# Re-prove it on THIS machine, exactly as a device does before trusting the pack:
ferralloy verify-cert reach.fpack
# → CERTIFIED — a device would trust this pack's correctness.
```

The gate has teeth: corrupt any weight in `cert.json` (as a flipped byte would) and
`ferralloy build`/`verify-cert` **reject** the pack, naming the box where `ΔV + α‖e‖² ≥ 0`.
The same re-proof runs inside the device agent (`ferralloyd`) before a pushed pack goes live,
and gates `ferralloy deploy` / `ferralloy release` — the operator promotion hook.

`policy/policy.wasm` here is a placeholder payload; in a real pack it is the control policy the
certificate is a guarantee *about*. The certificate re-verifier is dependency-free f64 + `tanh`,
so it runs unchanged from a browser (wasm32) to a Jetson to an MCU-class edge board.

**Cost.** The re-verification is cheap enough to run on the device: measured in-process (release,
`cargo run -p ferralloy-pack --example bench_cert`), a full re-proof takes **0.08 ms** (linear DI)
to **1.5 ms** (the nonlinear Van der Pol) — paid once, when a pack is accepted, not per control
step. The ~20 ms `ferralloy verify-cert` wall time is process startup + fpack load, not the verifier.

## The system registry — and what the two examples show

The verifier is a **registry of certified systems** (`system` field), not one hardcoded plant. Two
ship today, and together they demonstrate the law behind the energy:

| `cert.json` | system | energy | dynamics | result |
|---|---|---|---|---|
| `cert-linear-di.json` | `linear-double-integrator` | **empty head** — pure quadratic | affine, 1 case | R = 2.0, depth 0 — a smooth (convex-ROA) system needs no head |
| `cert-saturated-di.json` | `saturated-double-integrator` | **trained** ternary head | affine, 3 cases | R = 1.0 — the plain quadratic is refuted at R≈0.8; a trained head extends the region |
| `cert.json` | `saturated-hybrid-wall-contact` | ternary head (8 units) | affine, 6 cases | R = 1.2, 1486 boxes — the head earns the region past where the quadratic is refuted |
| `cert-van-der-pol.json` | `reversed-van-der-pol` | ternary head (8 units) | **nonlinear** `(x²−1)x` | R = 1.3, 4854 boxes — a non-convex ROA certified on-device, matching the dReal ground truth |

That is the law made concrete: **a learned ternary head certifies more than a quadratic iff the
quadratic is insufficient** — a non-convex ROA (Van der Pol), a mode switch (hybrid), or an input
constraint (saturated DI). The smooth double integrator needs no head, and its `dlyap` quadratic
certifies immediately. The saturated DI is the cleanest learned-beats-quadratic demonstration: same
smooth plant, but saturation refutes the quadratic at R≈0.8 while a trained head reaches R=1.0.

The Van der Pol has a genuinely **nonlinear** vector field, so its Jacobian varies with state —
the verifier handles it by ranging the Jacobian over each box and adding the dynamics' own Hessian
term. Its dense weight matrix is exactly `scale·T` with T∈{−1,0,1}, so it is a ternary energy after
all. The on-device Taylor+CROWN pass certifies it to R=1.3 — the same radius dReal reached with a
full SMT solver, but here with only `tanh` and a box worklist (soundness confirmed against a
Monte-Carlo ground-truth max of −0.015).

**Scope.** The four systems span the spectrum the gate is meant to cover: smooth (quadratic
suffices), input-constrained (trained head), hybrid mode-switching, and nonlinear non-convex.
Adding an *affine* system is a small registry entry; a nonlinear one exercises the Jacobian-range +
dynamics-Hessian path the Van der Pol uses; an input-constrained one uses the train-a-ternary-head
recipe the saturated DI demonstrates. Every certificate here was soundness-checked against a
Monte-Carlo ground-truth max before shipping. SOS/dReal remain a build-time/fleet gate for cases
beyond the on-device Taylor+CROWN pass (they need SDP/SMT solvers).
