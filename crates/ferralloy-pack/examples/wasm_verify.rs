//! Proof that the on-device certificate re-verifier runs inside a wasm sandbox.
//!
//! Build:  cargo build -p ferralloy-pack --example wasm_verify --release --target wasm32-wasip1
//! Run:    wasmtime target/wasm32-wasip1/release/examples/wasm_verify.wasm
//!
//! This re-proves the flagship NONLINEAR certificate (reversed Van der Pol, non-convex ROA) with
//! nothing but arithmetic + tanh — the same code path `ferralloyd` runs on a device, here executing
//! in wasmtime. The true browser build (wasm32-unknown-unknown) uses `--no-default-features` to drop
//! the signing/archive deps (getrandom); the verifier itself needs neither.

use ferralloy_pack::{reverify, CertificateSpec, TernaryEnergy};

fn main() {
    let spec = CertificateSpec {
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
    };
    println!("re-verifying a NONLINEAR Lyapunov certificate inside a wasm sandbox…");
    match reverify(&spec) {
        Ok(rep) => {
            println!(
                "CERTIFIED — {} over region {:?}: {} boxes, depth {}, worst ΔV+α‖e‖² {:+.5} (< 0 ⇒ sound)",
                spec.system, spec.region, rep.boxes, rep.depth, rep.worst_bound
            );
            println!("the device would ACCEPT this pack — verified correctness, in the browser/edge sandbox.");
        }
        Err(e) => {
            println!("REJECTED — {e}");
            std::process::exit(1);
        }
    }
}
