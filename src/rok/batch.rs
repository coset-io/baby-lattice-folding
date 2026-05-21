use rand::Rng;

use crate::{
    mat::Mat,
    relations::{LinInstance, LinRelation, LinWitness},
    ring::Rq,
};

/// Batch evaluation statements into less statements.
///     E.g. \tilde f(r)      = s
///     \tilde f(\bar r) = \bar s
///     -> tilde f(r) + c \tilde f(\bar r) = s + c \bar s
/// TODO: Use batch+ in p.34 to reduce n further. Needs another sumcheck.
pub fn rok_batch<const Q: u64, const D: usize>(
    lin: &LinRelation<Q, D>,
    n_target_eval_rows: usize,
    rng: &mut impl Rng,
) -> LinRelation<Q, D> {
    let h = &lin.instance.h;
    let f_com = &lin.instance.f_com;
    let f_eval = &lin.instance.f_eval;
    let y = &lin.instance.y;
    let n_top = lin.n_top();
    let n_orig_rows = lin.n_hat() - n_top;

    //
    // Verifier
    //
    let c = Rq::<Q, D>::random(rng);
    let c_matrix = Mat::<Rq<Q, D>>::from_fn(n_target_eval_rows, n_orig_rows, |i, j| {
        c.pow((i * n_orig_rows + j) as u64)
    });

    // Send `c_matrix` to Prover

    //
    // Both
    //
    // Calculate H_tilde = [H_top            ]
    //                     [c_matrix * H_bot]
    let h_top = h.submatrix(0..n_top, 0..h.ncols());
    let h_bot = h.submatrix(n_top..h.nrows(), 0..h.ncols());
    assert_eq!(h_bot.nrows(), n_orig_rows);
    let new_h_bot = c_matrix.clone() * h_bot;
    let h_tilde = h_top.stack(&new_h_bot);

    // Calculate Y_tilde = [Y_top           ]
    //                     [c_matrix * Y_bot]
    // Y_top = Y[:n_top]
    let y_top = y.submatrix(0..n_top, 0..y.ncols());
    assert_eq!(h_top.nrows(), y_top.nrows());
    // Y_bot = Y[n_top:]
    let y_bot = y.submatrix(n_top..y.nrows(), 0..y.ncols());
    assert_eq!(y_bot.nrows(), n_orig_rows);
    // Y_tilde = Y_top.stack(c_matrix * Y_bot)
    let y_tilde = y_top.stack(&(c_matrix * y_bot));
    assert_eq!(h_tilde.nrows(), y_tilde.nrows());

    // \beta = max(\beta_1, \beta_2)
    let new_instance =
        LinInstance::new(h_tilde, f_com.clone(), f_eval.clone(), y_tilde, lin.beta());
    let new_witness = LinWitness::new(lin.witness.w.clone());
    LinRelation::new(new_instance, new_witness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;
    use crate::relations::{LinInstance, LinWitness};
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

    /// rok_batch should collapse n_orig rows of H's bottom block down to
    /// `n_target_eval_rows`, leaving the top (F_com) block untouched.
    #[test]
    fn test_batch_n_hat_becomes_n_top_plus_target() {
        let rel = build_rel(
            mat(&[[1, 2]]),                 // F_com: 1 × 2 → n_top = 1
            mat(&[[3, 4], [5, 6], [7, 8]]), // F_eval: 3 × 2 → n_orig = 3
            mat(&[[1], [1]]),               // W: 2 × 1
            10_000,
        );
        assert_eq!(rel.n_hat(), 4); // 1 + 3 before batching
        let mut rng = rand::rng();
        let batched = rok_batch(&rel, 1, &mut rng);
        assert_eq!(batched.n_hat(), 2, "n_hat = n_top + n_target_eval_rows");
        assert_eq!(batched.n_top(), 1);
    }

    #[test]
    fn test_batch_collapse_to_zero_target_eval_rows() {
        // Edge: n_target = 0 ⇒ output H has only the F_com block.
        let rel = build_rel(
            mat(&[[1, 2]]),
            mat(&[[3, 4], [5, 6]]),
            mat(&[[1], [1]]),
            10_000,
        );
        let mut rng = rand::rng();
        let batched = rok_batch(&rel, 0, &mut rng);
        assert_eq!(batched.n_hat(), rel.n_top());
    }

    #[test]
    fn test_batch_to_one_eval_row_common_case() {
        // The typical use: many eval rows collapsed to one.
        let rel = build_rel(
            mat(&[[1, 2, 3]]),
            mat(&[[4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]),
            mat(&[[1], [1], [1]]),
            10_000,
        );
        let mut rng = rand::rng();
        let batched = rok_batch(&rel, 1, &mut rng);
        assert_eq!(batched.n_hat(), rel.n_top() + 1);
    }

    // ─── invariants preserved ───

    #[test]
    fn test_batch_preserves_f_com_and_f_eval() {
        // Batch only touches H and Y. F_com / F_eval are unchanged.
        let rel = build_rel(
            mat(&[[1, 2]]),
            mat(&[[3, 4], [5, 6], [7, 8]]),
            mat(&[[1], [1]]),
            10_000,
        );
        let mut rng = rand::rng();
        let batched = rok_batch(&rel, 1, &mut rng);
        assert_eq!(batched.instance.f_com, rel.instance.f_com);
        assert_eq!(batched.instance.f_eval, rel.instance.f_eval);
    }

    #[test]
    fn test_batch_preserves_witness_beta_m_r() {
        let rel = build_rel(
            mat(&[[1, 2]]),
            mat(&[[3, 4], [5, 6], [7, 8]]),
            mat(&[[1], [1]]),
            42,
        );
        let mut rng = rand::rng();
        let batched = rok_batch(&rel, 1, &mut rng);
        assert_eq!(batched.witness.w, rel.witness.w, "W unchanged");
        assert_eq!(batched.beta(), 42);
        assert_eq!(batched.m(), rel.m());
        assert_eq!(batched.r(), rel.r());
    }

    // ─── relation correctness ───

    /// rok_batch MUST produce a `LinRelation` whose `H_new · F · W = Y_new`
    /// holds for the verifier-sampled c_matrix. `LinRelation::new` panics if
    /// not, so reaching the end without panic confirms the algebraic identity.
    #[test]
    fn test_batch_produces_valid_relation() {
        let rel = build_rel(
            mat(&[[1, 2]]),
            mat(&[[3, 4], [5, 6], [7, 8]]),
            mat(&[[1], [1]]),
            10_000,
        );
        let mut rng = rand::rng();
        let _batched = rok_batch(&rel, 1, &mut rng);
    }

    #[test]
    fn test_batch_valid_when_n_target_equals_n_orig() {
        // n_target == n_orig: c_matrix is square. Algebra still must hold.
        let rel = build_rel(
            mat(&[[1, 2]]),
            mat(&[[3, 4], [5, 6]]),
            mat(&[[1], [1]]),
            10_000,
        );
        let mut rng = rand::rng();
        let _batched = rok_batch(&rel, 2, &mut rng);
    }
}
