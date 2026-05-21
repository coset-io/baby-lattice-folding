//! b-ary decomposition of a witness matrix into low-norm pieces.

use crate::{mat::Mat, relations::LinRelation, ring::Rq, zq::Zq};

/// ℓ = ⌈log_b(2β + 1)⌉ — number of base-b digits needed to encode the range
/// [-β, β]. Panics on β = 0 (range is degenerate).
pub fn get_l(_beta: u64, _b: u64) -> usize {
    todo!()
}

/// Balanced b-ary decomposition of a Z_q element into ℓ digits in [-⌊b/2⌋, ⌊b/2⌋].
///
/// E.g. b = 2:  7  → [ 1,  1,  1,  0, ...]
///              -7 → [-1, -1, -1,  0, ...]
/// E.g. b = 3:  5  → [-1, -1,  1,  0, ...]
///
/// Strategy: take the centered representative, peel off digits via repeated
/// mod-b. If a digit exceeds b/2, subtract b from it and carry +b into the
/// remaining value, keeping every digit balanced.
pub fn balanced_b_ary_decompose_zq<const Q: u64>(_f: Zq<Q>, _b: u64, _l: usize) -> Vec<Zq<Q>> {
    todo!()
}

/// Inverse of `balanced_b_ary_decompose_zq`: Σ_i coeffs[i] · b^i.
pub fn compose_zq<const Q: u64>(_coeffs: &[Zq<Q>], _b: u64) -> Zq<Q> {
    todo!()
}

/// Decompose witness W into ℓ matrices V_0, ..., V_{ℓ-1} such that
/// W = Σ_k b^k · V_k, with each V_k's polynomial coefficients in [-⌊b/2⌋, ⌊b/2⌋].
///
/// Per-entry decomposition runs `balanced_b_ary_decompose_zq` on every
/// coefficient of every R_q entry of W:
///   r = 4 + 5x + 3x^2  →  for each coefficient c at exponent `exp`:
///     c·x^{exp} = (d_0·b^0 + d_1·b^1 + ...) · x^{exp}
///              = d_0·b^0·x^{exp}  +  d_1·b^1·x^{exp}  +  ...
///                  V_0                 V_1            ...
pub fn decompose_w<const Q: u64, const D: usize>(
    _w: &Mat<Rq<Q, D>>,
    _b: u64,
    _l: usize,
) -> Vec<Mat<Rq<Q, D>>> {
    todo!()
}

/// Π^b-decomp: decomposes the witness W into ℓ low-norm chunks (V_0, ..., V_{ℓ-1})
/// and widens (W, Y) into (Ŵ, Ŷ) = (V_0 | ... | V_{ℓ-1}, Z_0 | ... | Z_{ℓ-1})
/// where Z_k = H · F · V_k.
///
/// Effect: H, F_com, F_eval, m, n, n̂ preserved; r grows r → ℓ·r; β tightens
/// from the per-entry centered bound.
pub fn rok_decompose<const Q: u64, const D: usize>(
    lin: &LinRelation<Q, D>,
    b: u64,
) -> LinRelation<Q, D> {
    let beta = lin.beta();
    let l = get_l(beta, b);

    //
    // Prover
    //
    let h = &lin.instance.h;
    let f = lin.instance.f();
    let w = &lin.witness.w;
    // Vs = decompose_w(W, b, l)
    // Zs = [H * F * V_k for V_k in Vs]

    // V_tilde = [V_0 || ... || V_{l-1}]
    // Z_tilde = [Z_0 || ... || Z_{l-1}]

    //
    // Verifier
    //
    let y = &lin.instance.y;
    // Y ?= Σ_{i=0}^{l-1} b^i · Z_i  — verifier recomputes and checks.

    //
    // Both
    //
    // Per-coefficient bound after balanced b-ary decomp is [-b/2, b/2].
    //   column ℓ_2^2  <=  m · d · (b//2)^2
    //   β            <=  ⌊b/2⌋ · √(m · d)
    // Uses isqrt so it stays integer (floor when m·d is not a square).
    // new_beta = (b // 2) * isqrt(m * d)
    let _ = (l, h, f, w, y, b);

    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;
    use crate::ring::Rq;
    use crate::zq::Zq;

    const Q: u64 = 17;
    const D: usize = 4;
    type F = Zq<Q>;
    type R = Rq<Q, D>;

    /// Z_q element from a signed integer (handles negatives via centered repr).
    fn zq(i: i64) -> F {
        let q = Q as i64;
        let v = i.rem_euclid(q) as u64;
        F::new(v)
    }

    /// Constant polynomial of value `v` in R_q (other coefficients zero).
    fn c(v: u64) -> R {
        let mut coeffs = [F::zero(); D];
        coeffs[0] = F::new(v);
        R::new(coeffs)
    }

    /// Monomial v·x^exp in R_q.
    fn mono(v: i64, exp: usize) -> R {
        assert!(exp < D);
        let mut coeffs = [F::zero(); D];
        coeffs[exp] = zq(v);
        R::new(coeffs)
    }

    /// Build a `Mat<R>` of constant-polynomial entries from u64 rows.
    fn mat<const N: usize>(rows: &[[u64; N]]) -> Mat<R> {
        let v: Vec<Vec<R>> = rows
            .iter()
            .map(|row| row.iter().map(|&v| c(v)).collect())
            .collect();
        Mat::new(v)
    }

    /// Build a valid LinRelation with H = I_{n_top + n_eval} and
    /// Y = H · F · W (so the relation invariant holds by construction).
    fn build_rel(f_com: Mat<R>, f_eval: Mat<R>, w: Mat<R>, beta: u64) -> LinRelation<Q, D> {
        let n_total = f_com.nrows() + f_eval.nrows();
        let h = Mat::<R>::identity(n_total);
        let f = f_com.stack(&f_eval);
        let y = h.clone() * f * w.clone();
        let inst = LinInstance::new(h, f_com, f_eval, y, beta);
        let wit = LinWitness::new(w);
        LinRelation::new(inst, wit)
    }

    // ─── get_l ───

    /// ℓ = number of base-b digits needed for the range [-β, β].
    #[test]
    fn test_get_l_explicit() {
        // β=1, b=2: 2β+1 = 3, need 2 binary digits.
        assert_eq!(get_l(1, 2), 2);
        // β=4, b=2: 2β+1 = 9, need 4 binary digits.
        assert_eq!(get_l(4, 2), 4);
        // β=7, b=3: 2β+1 = 15, need 3 ternary digits.
        assert_eq!(get_l(7, 3), 3);
    }

    /// β = 0 is a degenerate range; should panic.
    #[test]
    #[should_panic]
    fn test_get_l_beta_zero_panics() {
        let _ = get_l(0, 2);
    }

    // ─── balanced b-ary decomposition (Z_q-level) ───

    /// Concrete digit lists for documented cases (mirrors `test_decompose_Fq_explicit`).
    #[test]
    fn test_balanced_decompose_zq_explicit() {
        // 7 = 1 + 2 + 4 = 0b0111 → [1, 1, 1, 0]
        assert_eq!(
            balanced_b_ary_decompose_zq::<Q>(zq(7), 2, 4),
            vec![zq(1), zq(1), zq(1), zq(0)]
        );
        // Sign carries through: -7 → [-1, -1, -1, 0]
        assert_eq!(
            balanced_b_ary_decompose_zq::<Q>(zq(-7), 2, 4),
            vec![zq(-1), zq(-1), zq(-1), zq(0)]
        );
        // Zero → all zeros.
        assert_eq!(
            balanced_b_ary_decompose_zq::<Q>(zq(0), 2, 4),
            vec![zq(0); 4]
        );
        // Balanced ternary stress test: 5 with b=3 exercises the carry step.
        // Non-balanced would give [2, 1] (digit 2 ∉ {-1, 0, 1}); balanced uses
        // carry to push 2 → -1 with +3 added to the next position:
        //   5 = (-1)·1 + (-1)·3 + 1·9 → [-1, -1, 1]
        assert_eq!(
            balanced_b_ary_decompose_zq::<Q>(zq(5), 3, 3),
            vec![zq(-1), zq(-1), zq(1)]
        );
        // Sign symmetry for the same case.
        assert_eq!(
            balanced_b_ary_decompose_zq::<Q>(zq(-5), 3, 3),
            vec![zq(1), zq(1), zq(-1)]
        );
    }

    /// Reverse direction: `compose_zq` reassembles the digit lists above.
    #[test]
    fn test_compose_zq_explicit() {
        assert_eq!(compose_zq::<Q>(&[zq(1), zq(1), zq(1), zq(0)], 2), zq(7));
        assert_eq!(compose_zq::<Q>(&[zq(-1), zq(-1), zq(-1), zq(0)], 2), zq(-7));
        assert_eq!(compose_zq::<Q>(&[zq(0); 4], 2), zq(0));
        // Balanced ternary recompose: [-1, -1, 1] · (1, 3, 9) = -1 - 3 + 9 = 5.
        assert_eq!(compose_zq::<Q>(&[zq(-1), zq(-1), zq(1)], 3), zq(5));
        assert_eq!(compose_zq::<Q>(&[zq(1), zq(1), zq(-1)], 3), zq(-5));
    }

    /// compose(decompose(f)) == f for all f ∈ [-β, β] and several (b, β).
    #[test]
    fn test_decompose_zq_roundtrip() {
        for b in [2u64, 3] {
            for beta in [1u64, 4, 7] {
                let l = get_l(beta, b);
                for f_int in -(beta as i64)..=(beta as i64) {
                    let f = zq(f_int);
                    let coeffs = balanced_b_ary_decompose_zq::<Q>(f, b, l);
                    assert_eq!(
                        coeffs.len(),
                        l,
                        "decompose_zq must return ℓ={l} digits, got {} (f={f_int}, b={b})",
                        coeffs.len(),
                    );
                    let f_back = compose_zq::<Q>(&coeffs, b);
                    assert_eq!(
                        f_back, f,
                        "roundtrip mismatch: f={f_int}, b={b}, β={beta}, coeffs={coeffs:?}",
                    );
                }
            }
        }
    }

    // ─── decompose_w (matrix-level) ───

    /// W = Σ_k b^k · V_k where V = decompose_w(W, b, ℓ).
    #[test]
    fn test_decompose_w_roundtrip() {
        // W with mixed-degree polys; max coeff magnitude 3 → β=4 is safe.
        let w: Mat<R> = Mat::new(vec![
            vec![c(1) + mono(2, 1), c(3)],
            vec![R::zero(), c(0) - mono(1, 2)],
        ]);
        let b = 2u64;
        let beta = 4u64;
        let l = get_l(beta, b);
        let v = decompose_w(&w, b, l);

        // Shape: ℓ matrices, each same dim as W.
        assert_eq!(v.len(), l, "expected ℓ={l} matrices");
        for (k, v_k) in v.iter().enumerate() {
            assert_eq!(v_k.nrows(), w.nrows(), "V_{k} row count");
            assert_eq!(v_k.ncols(), w.ncols(), "V_{k} col count");
        }

        // Round-trip: Σ_k b^k · V_k must reassemble to W.
        // (Mat lacks scalar-mul; do it cell-wise.)
        let w_back = Mat::<R>::from_fn(w.nrows(), w.ncols(), |i, j| {
            let mut acc = R::zero();
            for (k, v_k) in v.iter().enumerate() {
                let bk = c(b.pow(k as u32));
                acc = acc + bk * v_k.row(i)[j];
            }
            acc
        });
        assert_eq!(w_back, w, "Σ_k b^k · V_k must equal W");
    }

    /// Every coefficient of every V_k lives in {-⌊b/2⌋, ..., ⌊b/2⌋}.
    /// For b=2, |coeff| ≤ 1 (i.e. coeff ∈ {-1, 0, 1}).
    #[test]
    fn test_decompose_w_norm_bound() {
        let w: Mat<R> = Mat::new(vec![vec![c(7), -mono(3, 1)], vec![mono(5, 2), R::zero()]]);
        let b = 2u64;
        let beta = 7u64;
        let l = get_l(beta, b);
        let v = decompose_w(&w, b, l);

        let bound = (b / 2) as i64;
        for (k, v_k) in v.iter().enumerate() {
            for i in 0..v_k.nrows() {
                for j in 0..v_k.ncols() {
                    for &coeff in v_k.row(i)[j].coeffs() {
                        let cv = coeff.to_centered().abs();
                        assert!(
                            cv <= bound,
                            "V_{k}[{i}][{j}] has |centered coeff|={cv} > ⌊b/2⌋={bound}",
                        );
                    }
                }
            }
        }
    }

    // ─── rok_decompose smoke ───

    /// Π^b-decomp: r grows by integer factor ℓ; H, F, m, n, n̂ unchanged.
    #[test]
    fn test_rok_decompose_smoke() {
        // β = 4, b = 2 → ℓ = ⌈log_2(9)⌉ = 4. r should grow r_in → 4·r_in.
        let rel = build_rel(
            mat(&[[1, 2]]),
            mat(&[[3, 4], [5, 6]]),
            mat(&[[1, 0], [0, 1]]),
            4,
        );
        let out = rok_decompose(&rel, 2);

        assert_eq!(out.m(), rel.m(), "m unchanged");
        assert_eq!(out.n(), rel.n(), "n unchanged");
        assert_eq!(out.n_hat(), rel.n_hat(), "n̂ unchanged");
        assert_eq!(out.n_top(), rel.n_top(), "n_top unchanged");
        assert_eq!(out.instance.f_com, rel.instance.f_com, "F_com preserved");
        assert_eq!(out.instance.f_eval, rel.instance.f_eval, "F_eval preserved");
        assert_eq!(out.instance.h, rel.instance.h, "H preserved");

        assert!(out.r() > rel.r(), "r must strictly grow");
        assert_eq!(out.r() % rel.r(), 0, "r grows by integer factor ℓ");
        let ell = out.r() / rel.r();
        assert!(ell > 1, "ℓ > 1 for non-trivial decomposition");

        // β should not grow (per-entry bound tightens).
        assert!(out.beta() <= rel.beta(), "β must not grow");
    }

    /// rok_decompose MUST produce a `LinRelation` whose `H · F · Ŵ = Ŷ` holds.
    /// Reaching the end without panic = LinRelation::new's invariant check passed.
    #[test]
    fn test_rok_decompose_produces_valid_relation() {
        let rel = build_rel(
            mat(&[[1, 2]]),
            mat(&[[3, 4], [5, 6]]),
            mat(&[[1, 0], [0, 1]]),
            4,
        );
        let _out = rok_decompose(&rel, 2);
    }
}
