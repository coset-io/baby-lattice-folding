//! Π^fold: folds r linear relations into one via a small ternary challenge.

use rand::Rng;

use crate::{
    mat::Mat,
    relations::{LinInstance, LinRelation, LinWitness},
    ring::Rq,
    zq::Zq,
};

/// Sample a single Rq element with ternary {-1, 0, 1} coefficients, p = 1/3 each.
/// Different from the χ used in rok_rp (p(0)=1/2, p(±1)=1/4).
fn sample_small_rq<const Q: u64, const D: usize>(rng: &mut impl Rng) -> Rq<Q, D> {
    let coeffs: [Zq<Q>; D] = std::array::from_fn(|_| Zq::<Q>::random_low_norm(rng));
    Rq::new(coeffs)
}

/// Sample challenge matrix C ∈ R_q^{r_in × r_out} with ternary entries.
fn sample_c<const Q: u64, const D: usize>(
    r_in: usize,
    r_out: usize,
    rng: &mut impl Rng,
) -> Mat<Rq<Q, D>> {
    // challenge set: larger challenge set over subtractive set
    // rok_rp computes and proves about randomised projections of the witness.
    // This allows us to use a much larger challenge set in the folding step Π fold
    // instead of a subtractive set (used in [KLNO24]), which ultimately removes (roughly) one λ factor from the proof size of [KLNO24]

    // SALSAA paper p.36
    // We take a different approach and sample challenges so that the coefficients
    // are sampled uniformly from the ternary set {−1, 0, 1}. The set of challenges
    // with ternary coefficients is not strong samplings sets per definition,
    // but the probability of sampling elements so that the inverse of two is non-invertible is small.
    Mat::<Rq<Q, D>>::from_fn(r_in, r_out, |_i, _j| sample_small_rq(rng))
}

/// Π^fold: collapses the r witness columns of `lin` into `r_out` by sampling a
/// random ternary challenge matrix C ∈ R_q^{r_in × r_out} and computing
/// (W, Y) ↦ (W·C, Y·C). H, F_com, F_eval are preserved.
pub fn rok_fold<const Q: u64, const D: usize>(
    lin: &LinRelation<Q, D>,
    r_out: usize,
    rng: &mut impl Rng,
) -> LinRelation<Q, D> {
    let r_in = lin.r();

    //
    // Verifier
    //
    // C = sample_C(r_in, r_out)
    let c = sample_c::<Q, D>(r_in, r_out, rng);
    // Y_tilde = Y * C
    let y = &lin.instance.y;
    let y_tilde = y.clone() * c.clone();

    //
    // Prover
    //
    let h = &lin.instance.h;
    let f_com = &lin.instance.f_com;
    let f_eval = &lin.instance.f_eval;
    let w = &lin.witness.w;
    // \tilde W = W * C
    let w_tilde = w.clone() * c;

    //
    // Check relation holds
    //
    // Derive new norm bound: every entry of C is ternary {-1, 0, 1}; viewed as an Rq element with d
    // ternary coefficients:
    //   |C_ij · W_i|       <= d · |W_i|   = d · β
    //   |Σ_i C_ij · W_i|  <= r_in · d · β
    let new_beta = (r_in * D) as u64 * lin.beta();

    // H * F * \tilde W = \tilde Y
    LinRelation::new(
        LinInstance::new(h.clone(), f_com.clone(), f_eval.clone(), y_tilde, new_beta),
        LinWitness::new(w_tilde),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;
    use crate::ring::Rq;
    use crate::zq::Zq;

    const Q: u64 = 17;
    const D: usize = 4;
    type R = Rq<Q, D>;

    /// Constant polynomial of value `v` in R_q (other coefficients zero).
    fn c(v: u64) -> R {
        let mut coeffs = [Zq::<Q>::zero(); D];
        coeffs[0] = Zq::<Q>::new(v);
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

    // ─── shape / dim correctness ───

    /// Π^fold collapses witness width r → r_out; H, F, m, n, n̂ all preserved.
    #[test]
    fn test_fold_collapses_r_to_r_out() {
        let rel = build_rel(
            mat(&[[1, 2]]),         // F_com: 1 × 2 → n_top = 1
            mat(&[[3, 4], [5, 6]]), // F_eval: 2 × 2
            mat(&[[1, 0], [0, 1]]), // W: 2 × 2 → r_in = 2
            10_000,
        );
        assert_eq!(rel.r(), 2);
        let mut rng = rand::rng();
        let folded = rok_fold(&rel, 1, &mut rng);

        assert_eq!(folded.r(), 1, "r collapsed to r_out");
        assert_eq!(folded.n_hat(), rel.n_hat(), "n̂ unchanged");
        assert_eq!(folded.n(), rel.n(), "n unchanged");
        assert_eq!(folded.m(), rel.m(), "m unchanged");
        assert_eq!(folded.n_top(), rel.n_top(), "n_top unchanged");
    }

    // ─── invariants preserved ───

    /// Fold only touches W and Y; F_com / F_eval / H stay identical.
    #[test]
    fn test_fold_preserves_h_f_com_and_f_eval() {
        let rel = build_rel(
            mat(&[[1, 2]]),
            mat(&[[3, 4], [5, 6]]),
            mat(&[[1, 0], [0, 1]]),
            10_000,
        );
        let mut rng = rand::rng();
        let folded = rok_fold(&rel, 1, &mut rng);

        assert_eq!(folded.instance.f_com, rel.instance.f_com);
        assert_eq!(folded.instance.f_eval, rel.instance.f_eval);
        assert_eq!(folded.instance.h, rel.instance.h);
    }

    /// β_new = r_in · d · β_old (worst-case norm of ternary linear combination).
    #[test]
    fn test_fold_beta_growth_bound() {
        let rel = build_rel(
            mat(&[[1, 2]]),
            mat(&[[3, 4], [5, 6]]),
            mat(&[[1, 0], [0, 1]]),
            5,
        );
        let mut rng = rand::rng();
        let folded = rok_fold(&rel, 1, &mut rng);
        let expected = (rel.r() as u64) * (D as u64) * rel.beta();
        assert_eq!(folded.beta(), expected, "β_new = r_in · d · β_old");
    }

    // ─── relation correctness ───

    /// rok_fold MUST produce a `LinRelation` whose `H · F · Ŵ = Ŷ` holds.
    /// LinRelation::new panics if not, so reaching the end without panic
    /// confirms the folded relation is internally consistent.
    #[test]
    fn test_fold_produces_valid_relation() {
        let rel = build_rel(
            mat(&[[1, 2]]),
            mat(&[[3, 4], [5, 6]]),
            mat(&[[1, 0], [0, 1]]),
            10_000,
        );
        let mut rng = rand::rng();
        let _folded = rok_fold(&rel, 1, &mut rng);
    }

    /// r_out == r_in: C is square, fold is a (random) re-mixing of columns.
    /// Algebraic identity must still hold.
    #[test]
    fn test_fold_r_out_equals_r_in() {
        let rel = build_rel(
            mat(&[[1, 2]]),
            mat(&[[3, 4], [5, 6]]),
            mat(&[[1, 0], [0, 1]]),
            10_000,
        );
        let mut rng = rand::rng();
        let folded = rok_fold(&rel, 2, &mut rng);
        assert_eq!(folded.r(), 2);
    }
}
