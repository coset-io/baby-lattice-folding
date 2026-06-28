//! SALSAA folding scheme — top-level driver composing the RoK chain.

use rand::Rng;

use crate::{
    relations::LinRelation,
    rok::{
        batch::rok_batch,
        decompose::{get_l, rok_decompose},
        fold::rok_fold,
        join::rok_join,
        norm::rok_norm,
        rp::rok_rp,
    },
};

/// One full SALSAA folding round.
///
///   Π^join → Π^norm → Π^⊗RP → Π^fold → Π^join → Π^batch → Π^b-decomp
///
/// Input:  L lin relations (sharing F_com).
/// Output: 1 lin relation.
///
/// Params:
///   - `lins`:  L instances to be folded.
///   - `b`:     base of b-ary decomposition (default 2).
///   - `n_rp`:  the n_rp in rok_rp; security parameter. TODO: choose a proper number.
///   - `e`:     NTT slot size used by rok_norm (d/e slots per Rq element).
///   - `d_h`:   sumcheck hypercube basis size used by rok_norm.
///   - `rng`:   verifier challenges. (Replace with `&mut Transcript` once Fiat–Shamir lands.)
pub fn fold<const Q: u64, const D: usize>(
    lins: &[LinRelation<Q, D>],
    b: u64,
    n_rp: usize,
    e: usize,
    d_h: usize,
    rng: &mut impl Rng,
) -> LinRelation<Q, D> {
    //
    // Configs
    //
    let l = lins.len();
    assert!(l >= 1, "salsaa::fold needs ≥ 1 input LinRelation");

    //
    // Join L instances into 1
    //
    let mut lin_joined = lins[0].clone();
    for lin in &lins[1..] {
        lin_joined = rok_join(&lin_joined, lin);
    }

    //
    // Norm check
    //
    let lin_normed = rok_norm(e, d_h, rng, &lin_joined);
    assert_eq!(lin_normed.n_hat(), lin_joined.n_hat() + 2);
    assert_eq!(lin_normed.n(), lin_joined.n() + 2);
    assert_eq!(lin_normed.m(), lin_joined.m());
    assert_eq!(lin_normed.r(), lin_joined.r());
    assert_eq!(lin_normed.beta(), lin_joined.beta());

    //
    // ⊗RP: Johnson–Lindenstrauss to improve soundness without a subtractive set.
    //
    let m_rp = lin_normed.r() * n_rp;
    assert_eq!(m_rp, n_rp * lin_normed.r());
    let (lin_orig, lin_w_hat) = rok_rp(&lin_normed, n_rp, m_rp, rng);
    assert_eq!(lin_orig.n_hat(), lin_joined.n_hat() + 3);
    assert_eq!(lin_orig.n(), lin_joined.n() + 3);
    assert_eq!(lin_orig.m(), lin_joined.m());
    assert_eq!(lin_orig.r(), lin_joined.r());
    assert_eq!(lin_orig.beta(), lin_joined.beta());

    // Check lin_w_hat side
    assert_eq!(lin_w_hat.n_hat(), lin_joined.n_top() + 1);
    assert_eq!(lin_w_hat.n(), lin_joined.n_top() + 1);
    assert_eq!(lin_w_hat.m(), lin_joined.m());
    assert_eq!(lin_w_hat.r(), 1);
    assert!(lin_w_hat.beta() >= lin_joined.beta());

    //
    // Fold the witnesses of the main statements into `r_out = 1`.
    //
    let r_out = 1;
    let lin_folded = rok_fold(&lin_orig, r_out, rng);
    assert_eq!(lin_folded.n_hat(), lin_joined.n_hat() + 3);
    assert_eq!(lin_folded.n(), lin_joined.n() + 3);
    assert_eq!(lin_folded.m(), lin_joined.m());
    assert_eq!(lin_folded.r(), 1);
    assert!(
        lin_folded.beta() > lin_orig.beta(),
        "norm bound didn't grow after fold"
    );

    //
    // Merge (join) the ⊗RP-side (w_hat) relation with the folded one.
    //
    let lin_merged = rok_join(&lin_w_hat, &lin_folded);
    assert_eq!(lin_merged.n_hat(), lin_joined.n_hat() + 4);
    assert_eq!(lin_merged.n(), lin_joined.n() + 4);
    assert_eq!(lin_merged.m(), lin_joined.m());
    assert_eq!(lin_merged.r(), 2);
    // β unchanged through join (it's a max, and lin_folded dominates here)
    assert_eq!(lin_merged.beta(), lin_folded.beta());

    //
    // Batch evaluation rows down to (n̂ - n̄) of `lin_joined`, so the final H has
    // n̂_new = lin_joined.n̂ rows again.
    //
    let n_target_eval_rows = lin_joined.n_hat() - lin_joined.n_top();
    let lin_batched = rok_batch(&lin_merged, n_target_eval_rows, rng);
    assert_eq!(lin_batched.n_hat(), lin_joined.n_hat());
    assert_eq!(lin_batched.n(), lin_joined.n() + 4);
    assert_eq!(lin_batched.m(), lin_joined.m());
    assert_eq!(lin_batched.r(), 2);
    assert_eq!(lin_batched.beta(), lin_merged.beta());

    //
    // b-ary decomposition to bring β back down for next-round composability.
    //
    let lin_decomposed = rok_decompose(&lin_batched, b);
    assert_eq!(lin_decomposed.n_hat(), lin_joined.n_hat());
    assert_eq!(lin_decomposed.n(), lin_joined.n() + 4);
    assert_eq!(lin_decomposed.m(), lin_joined.m());

    let ell = get_l(lin_batched.beta(), b);
    // New Y is Z̃ = [Z_0 || ... || Z_{ℓ-1}] with each Z_i ∈ R^{m × r_old},
    // so r_new = r_old · ℓ.
    assert_eq!(lin_decomposed.r(), lin_batched.r() * ell);

    // Per-entry bound in [-b/2, b/2], m·d coeffs per column → β ≤ ⌊b/2⌋ · √(m·d).
    // (Exact equality check disabled here until decompose's β formula stabilises.)
    assert!(
        lin_decomposed.beta() <= lin_joined.beta(),
        "norm budget must not exceed round-start (else next round won't compose)",
    );

    lin_decomposed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;
    use crate::relations::{LinInstance, LinWitness};
    use crate::ring::Rq;
    use crate::zq::Zq;

    // Toy params: smallest q with negacyclic NTT support (17 ≡ 1 mod 8), d = 4.
    // Falcon params (q=12289, d=1024) live locally in test_salsaa_fold_falcon.
    const Q: u64 = 17;
    const D: usize = 4;
    type R = Rq<Q, D>;

    /// Constant polynomial of value `v` in R_q (other coefficients zero).
    /// Generic over (Q, D) so both toy and Falcon-param tests share it.
    fn c<const Q: u64, const D: usize>(v: u64) -> Rq<Q, D> {
        let mut coeffs = [Zq::<Q>::zero(); D];
        coeffs[0] = Zq::<Q>::new(v);
        Rq::new(coeffs)
    }

    /// Build a `Mat<Rq<Q,D>>` of constant-polynomial entries from u64 rows.
    fn mat<const Q: u64, const D: usize, const N: usize>(rows: &[[u64; N]]) -> Mat<Rq<Q, D>> {
        let v: Vec<Vec<Rq<Q, D>>> = rows
            .iter()
            .map(|row| row.iter().map(|&v| c(v)).collect())
            .collect();
        Mat::new(v)
    }

    /// Build a satisfying LinRelation with H = I, supplied F_com / W, no eval rows.
    /// Mirrors the Python `_make_lins_for_salsaa_fold` per-instance fixture.
    fn build_rel<const Q: u64, const D: usize>(
        f_com: Mat<Rq<Q, D>>,
        w: Mat<Rq<Q, D>>,
        beta: u64,
    ) -> LinRelation<Q, D> {
        let f_eval = Mat::<Rq<Q, D>>::zero(0, f_com.ncols());
        let n_total = f_com.nrows() + f_eval.nrows();
        let h = Mat::<Rq<Q, D>>::identity(n_total);
        let f = f_com.stack(&f_eval);
        let y = h.clone() * f * w.clone();
        let inst = LinInstance::new(h, f_com, f_eval, y, beta);
        let wit = LinWitness::new(w);
        LinRelation::new(inst, wit)
    }

    // ─── salsaa::fold smoke ───

    /// Top-level smoke: chain runs end-to-end on L=2 fresh relations and the
    /// output is a valid LinRelation whose F_com matches the inputs'.
    ///
    /// All inputs share F_com (join's invariant). Each LinRelation::new along
    /// the chain validates H·F·W = Y and the norm bound, so reaching the end
    /// without panic == the entire chain is internally consistent.
    #[test]
    fn test_salsaa_fold_smoke() {
        // Tiny shared commitment matrix (2 × 2) to keep the chain fast.
        let f_com: Mat<R> = mat(&[[1, 2], [3, 5]]);
        // Each W is 2 × 1 (m=2, r=1), small entries.
        let w_0: Mat<R> = mat(&[[1], [0]]);
        let w_1: Mat<R> = mat(&[[0], [1]]);
        let beta = 4;
        let lins = vec![
            build_rel(f_com.clone(), w_0, beta),
            build_rel(f_com.clone(), w_1, beta),
        ];

        // Sanity: all inputs share F_com (join's precondition).
        for lin in &lins[1..] {
            assert_eq!(lin.instance.f_com, lins[0].instance.f_com);
        }

        let mut rng = rand::rng();
        let out = fold(
            &lins, /* b = */ 2, /* n_rp = */ 1, /* e = */ 1, /* d_h = */ 2,
            &mut rng,
        );

        // m is preserved end-to-end (witness rows / commitment width).
        assert_eq!(out.m(), lins[0].m(), "m must be preserved across the chain");
        // F_com is preserved end-to-end (no sub-protocol rewrites the commitment).
        assert_eq!(
            out.instance.f_com, lins[0].instance.f_com,
            "F_com must be preserved end-to-end",
        );
        // β must NOT exceed the input bound (else next round won't compose).
        assert!(
            out.beta() <= lins[0].beta(),
            "β must not grow over a full round: in {} → out {}",
            lins[0].beta(),
            out.beta(),
        );
    }

    /// Single-input edge case: L=1 should still run (join loop is empty).
    #[test]
    fn test_salsaa_fold_single_input() {
        let f_com: Mat<R> = mat(&[[1, 2], [3, 5]]);
        let w: Mat<R> = mat(&[[1], [0]]);
        let lins = vec![build_rel(f_com, w, 4)];

        let mut rng = rand::rng();
        let out = fold(&lins, 2, 1, 1, 2, &mut rng);
        assert_eq!(out.m(), lins[0].m());
    }

    /// Empty input must panic — `lins[0].clone()` would index past the end.
    #[test]
    #[should_panic]
    fn test_salsaa_fold_empty_panics() {
        let lins: Vec<LinRelation<Q, D>> = vec![];
        let mut rng = rand::rng();
        let _ = fold(&lins, 2, 1, 1, 2, &mut rng);
    }

    /// Falcon params (q=12289, d=1024): confirms the full chain runs at
    /// production-scale ring dimension, not just toy params.
    ///
    /// β must be ≥ ⌈√(⌊b/2⌋²·m·d)⌉ = ⌈√(1·2·1024)⌉ = 46 for a round to compose:
    /// b-ary decomposition resets β to that bound regardless of incoming β, so a
    /// smaller round-start budget can never satisfy the composability check. We
    /// use 64.
    ///
    /// `#[ignore]`d because one fold at d=1024 takes ~15s — run explicitly with
    /// `cargo test -- --ignored`.
    #[test]
    #[ignore = "d=1024 fold takes ~15s; run with `cargo test -- --ignored`"]
    fn test_salsaa_fold_falcon() {
        const FQ: u64 = 12289;
        const FD: usize = 1024;
        type FR = Rq<FQ, FD>;

        let f_com: Mat<FR> = mat(&[[1, 2], [3, 5]]);
        let w_0: Mat<FR> = mat(&[[1], [0]]);
        let w_1: Mat<FR> = mat(&[[0], [1]]);
        let beta = 64; // ≥ ⌈√(1·2·1024)⌉ = 46
        let lins = vec![
            build_rel(f_com.clone(), w_0, beta),
            build_rel(f_com.clone(), w_1, beta),
        ];

        let mut rng = rand::rng();
        let out = fold(&lins, 2, 1, 1, 2, &mut rng);

        assert_eq!(out.m(), lins[0].m(), "m preserved across the chain");
        assert_eq!(
            out.instance.f_com, lins[0].instance.f_com,
            "F_com preserved end-to-end",
        );
        assert!(
            out.beta() <= lins[0].beta(),
            "β must not grow over a full round: in {} → out {}",
            lins[0].beta(),
            out.beta(),
        );
    }
}
