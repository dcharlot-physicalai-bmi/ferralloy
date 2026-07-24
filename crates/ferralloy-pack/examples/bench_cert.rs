//! Micro-benchmark: the true in-process cost of on-device certificate re-verification.
//!
//! `ferralloy verify-cert` measures ~20 ms end-to-end, but that is dominated by process startup +
//! fpack load; a long-running `ferralloyd` agent calls `reverify()` in-process and pays only the
//! arithmetic. This times that arithmetic alone, per system.
//!
//! Run: `cargo run -p ferralloy-pack --example bench_cert --release`

use ferralloy_pack::{reverify, CertificateSpec, TernaryEnergy};
use std::time::Instant;

fn spec(system: &str, region: [f64; 2], p: [f64; 4], scale: f64, t: Vec<i8>, b1: Vec<f64>, w2: Vec<f64>, v0: f64) -> CertificateSpec {
    CertificateSpec {
        kind: "lyapunov-ternary-taylor-crown".into(),
        system: system.into(),
        region,
        alpha: 5e-4,
        energy: TernaryEnergy { p, scale, t, b1, w2, v0 },
    }
}

fn main() {
    let specs = vec![
        spec("linear-double-integrator", [0.15, 2.0],
            [84.186700371440466, 2.2125206355757552, 2.2125206355757534, 9.5185075319851435],
            1.0, vec![], vec![], vec![], 0.0),
        spec("saturated-double-integrator", [0.15, 1.0],
            [84.18670037144047, 2.212520635575755, 2.2125206355757534, 9.518507531985144], 1.9497924248377483,
            vec![0, -1, 0, -1, 0, 1, 0, -1, 0, -1, 0, 1, 0, 1, 1, -1],
            vec![-1.7654157876968384, -1.7829656600952148, -1.6165714263916016, 1.5079240798950195,
                 -1.8295671939849854, 2.0675222873687744, -1.7644506692886353, 1.109099268913269],
            vec![1.0797884464263916, 1.277887225151062, 1.6327425241470337, -1.4929447174072266,
                 1.0733451843261719, -1.2776927947998047, 1.4379184246063232, 1.4926438331604004],
            -7.501433930624267),
        spec("saturated-hybrid-wall-contact", [0.15, 1.2],
            [31.988, 2.543, 2.543, 1.4169999999999998], 1.4740514336487223,
            vec![-1, 0, -1, 0, 0, 0, -1, -1, -1, 0, 1, 1, 1, 0, 0, 0],
            vec![-1.7253050443315214, -1.6895583892614585, -1.572812794286765, -2.9962607818256326,
                 -1.362285297766529, 2.9948712315814445, 1.6641829682841895, 0.15496976355688338],
            vec![-1.8092778484049137, -0.6474111753241604, -0.0407591631059273, 1.2424339130389697,
                 -0.3175122724570142, -1.1023667519975417, 0.6676446220380198, -5.977279221172131e-12],
            0.9069008854718814),
        spec("reversed-van-der-pol", [0.15, 1.3],
            [76.2755359693619, -25.51278060966837, -25.51278060966837, 51.27806096683688], 3.4601597785949707,
            vec![-1, 0, 0, -1, 1, 0, 0, -1, 0, -1, -1, 0, 0, 1, 1, 0],
            vec![-2.3375589847564697, -2.102060556411743, -2.708591938018799, -2.0922701358795166,
                 -2.0625295639038086, 1.6640418767929077, -2.6110639572143555, -2.7082889080047607],
            vec![-7.458496570587158, 3.7638297080993652, -4.320842266082764, 3.884138584136963,
                 3.7923171520233154, 1.5532889366149902, 7.402223110198975, -4.4193596839904785],
            -0.9857819872128699),
    ];

    println!("in-process certificate re-verification (release):");
    println!("  {:<30} {:>8} {:>10} {:>12}", "system", "boxes", "ms/verify", "verifies/s");
    for s in &specs {
        // warm up + establish box count
        let rep = reverify(s).expect("must certify");
        // time enough iterations for a stable read
        let iters = if rep.boxes > 3000 { 40 } else { 80 };
        let t0 = Instant::now();
        for _ in 0..iters { let _ = reverify(s).unwrap(); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("  {:<30} {:>8} {:>10.3} {:>12.0}", s.system, rep.boxes, ms, 1000.0 / ms);
    }
    println!("\nThis is what a long-running ferralloyd agent pays per pack — the ~20 ms `verify-cert`");
    println!("wall time is process startup + fpack load, not the verifier.");
}
