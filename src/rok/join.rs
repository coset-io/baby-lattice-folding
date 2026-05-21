//! Combines two linear relations sharing the same commitment matrix into one.

use std::cmp::max;

use crate::{
    mat::Mat,
    relations::{LinInstance, LinRelation, LinWitness},
    ring::Rq,
};

pub fn rok_join<const Q: u64, const D: usize>(
    lin_0: &LinRelation<Q, D>,
    lin_1: &LinRelation<Q, D>,
) -> LinRelation<Q, D> {
    assert_eq!(lin_0.m(), lin_1.m());

    let f_0_com = &lin_0.instance.f_com;
    let f_0_eval = &lin_0.instance.f_eval;
    let h_0 = &lin_0.instance.h;
    let w_0 = &lin_0.witness.w;
    let y_0 = &lin_0.instance.y;

    let f_1_com = &lin_1.instance.f_com;
    let f_1_eval = &lin_1.instance.f_eval;
    let h_1 = &lin_1.instance.h;
    let w_1 = &lin_1.witness.w;
    let y_1 = &lin_1.instance.y;

    // Commitment must be the same
    assert_eq!(f_0_com, f_1_com);
    let n_top = f_0_com.nrows();

    //
    // Both
    //
    // Get the bottoms of Hs
    let h_0_bot = h_0.submatrix(n_top..h_0.nrows(), n_top..h_0.ncols());
    let h_1_bot = h_1.submatrix(n_top..h_1.nrows(), n_top..h_1.ncols());

    //
    // Prover
    //
    // FIXME: fix the matrix mul to support mul for refs, and get rid of these clones.
    let y_01_bot = h_0_bot.clone() * f_0_eval.clone() * w_1.clone();
    let y_10_bot = h_1_bot.clone() * f_1_eval.clone() * w_0.clone();
    // Send y_01_bot and y_10_bot to Verifier

    //
    // Both Prover and Verifier
    //

    // H_new = [I                    ]
    //         [     H_0_bot         ]
    //         [              H_1_bot]
    let h_new: Mat<Rq<Q, D>> =
        Mat::block_diagonal(&[Mat::<Rq<Q, D>>::identity(n_top), h_0_bot, h_1_bot]);

    // F = [F_top  ]
    //     [F_0_bot]
    //     [F_1_bot]
    let f_new_bot = f_0_eval.stack(f_1_eval);

    // Y = [
    //     [Y_0[:n_top], Y_1[:n_top]],
    //     [Y_0[n_top:], Y_01_bot],
    //     [Y_10_bot, Y_1[n_top:]],
    // ]
    let y_new_0 = y_0
        .submatrix(0..n_top, 0..y_0.ncols())
        .augment(&y_1.submatrix(0..n_top, 0..y_1.ncols()));
    let y_new_1 = y_0
        .submatrix(n_top..y_0.nrows(), 0..y_0.ncols())
        .augment(&y_01_bot);
    let y_new_2 = y_10_bot.augment(&y_1.submatrix(n_top..y_1.nrows(), 0..y_1.ncols()));
    let y_new = y_new_0.stack(&y_new_1).stack(&y_new_2);

    //
    // Check relation holds
    //
    // W = [W_0 | W_1]
    let w_new = w_0.augment(w_1);

    // \beta = max(\beta_1, \beta_2)
    let new_beta = max(lin_0.beta(), lin_1.beta());
    let new_instance = LinInstance::new(h_new, f_0_com.clone(), f_new_bot, y_new, new_beta);
    let new_witness = LinWitness::new(w_new);
    LinRelation::new(new_instance, new_witness)
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

    /// Build a valid LinRelation with H = I_n_total, the supplied F_com / F_eval / W,
    /// and Y = H · F · W (so the invariant always holds).
    fn build_rel(f_com: Mat<R>, f_eval: Mat<R>, w: Mat<R>, beta: u64) -> LinRelation<Q, D> {
        let n_total = f_com.nrows() + f_eval.nrows();
        let h = Mat::<R>::identity(n_total);
        let f = f_com.stack(&f_eval);
        let y = h.clone() * f * w.clone();
        let inst = LinInstance::new(h, f_com, f_eval, y, beta);
        let wit = LinWitness::new(w);
        LinRelation::new(inst, wit)
    }

    // ─── dim correctness ───

    #[test]
    fn test_join_dimensions() {
        // Both rels: m = 2, n_top = 1, n_eval = 1 → n = 2, r = 1.
        let f_com = mat(&[[1, 2]]);

        let rel_0 = build_rel(
            f_com.clone(),
            mat(&[[3, 4]]),   // F_eval_0
            mat(&[[1], [0]]), // W_0  (small norm so β is satisfied)
            10_000,
        );
        let rel_1 = build_rel(
            f_com.clone(),
            mat(&[[5, 6]]),   // F_eval_1
            mat(&[[0], [1]]), // W_1
            10_000,
        );

        let joined = rok_join(&rel_0, &rel_1);

        assert_eq!(joined.m(), 2, "m unchanged");
        assert_eq!(joined.n_top(), 1, "n_top = F_com.nrows() unchanged");
        // n̂_new = n_top + (n̂_0 - n_top) + (n̂_1 - n_top) = 1 + 1 + 1 = 3
        assert_eq!(joined.n_hat(), 3, "n̂_new = n̂_0 + n̂_1 - n_top");
        // r_new = r_0 + r_1
        assert_eq!(joined.r(), 2, "r_new = r_0 + r_1");
    }

    #[test]
    fn test_join_beta_is_max() {
        let f_com = mat(&[[1, 2]]);
        let rel_0 = build_rel(f_com.clone(), mat(&[[3, 4]]), mat(&[[1], [0]]), 42);
        let rel_1 = build_rel(f_com.clone(), mat(&[[5, 6]]), mat(&[[0], [1]]), 100);
        let joined = rok_join(&rel_0, &rel_1);
        assert_eq!(joined.beta(), 100);
    }

    // ─── correctness of joined relation (relies on LinRelation::new's invariant check) ───

    /// rok_join MUST produce a `LinRelation` whose `H · F · W = Y` holds.
    /// LinRelation::new panics if not, so this test simply running through
    /// without panic confirms the joined relation is internally consistent.
    #[test]
    fn test_join_produces_valid_relation() {
        let f_com = mat(&[[1, 2]]);
        let rel_0 = build_rel(f_com.clone(), mat(&[[3, 4]]), mat(&[[1], [0]]), 10_000);
        let rel_1 = build_rel(f_com.clone(), mat(&[[5, 6]]), mat(&[[0], [1]]), 10_000);
        let _joined = rok_join(&rel_0, &rel_1);
        // If we reach here, LinRelation::new inside rok_join didn't panic ⇒
        // H_new · F_new · W_new = Y_new and ‖W_new‖₂² ≤ β².
    }

    // ─── precondition: m must match ───

    #[test]
    #[should_panic]
    fn test_join_different_m_panics() {
        let rel_0 = build_rel(mat(&[[1, 2]]), mat(&[[3, 4]]), mat(&[[1], [0]]), 10_000);
        let rel_1 = build_rel(
            mat(&[[1, 2, 3]]),
            mat(&[[4, 5, 6]]),
            mat(&[[1], [0], [0]]),
            10_000,
        );
        let _ = rok_join(&rel_0, &rel_1);
    }

    // ─── precondition: F_com must match (commitment-sharing assumption) ───

    #[test]
    #[should_panic]
    fn test_join_different_f_com_panics() {
        // Both rels have m = 2 but different F_com contents.
        let rel_0 = build_rel(mat(&[[1, 2]]), mat(&[[3, 4]]), mat(&[[1], [0]]), 10_000);
        let rel_1 = build_rel(
            mat(&[[7, 8]]),
            mat(&[[5, 6]]), // different F_com from rel_0
            mat(&[[0], [1]]),
            10_000,
        );
        let _ = rok_join(&rel_0, &rel_1);
    }
}
