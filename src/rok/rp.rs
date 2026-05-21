//! Π^rp: Johnson–Lindenstrauss-style random projection that reduces
//! witness dimension via a sparse ternary challenge matrix.

use rand::Rng;

use crate::{mat::Mat, relations::LinRelation, ring::Rq};

/// Sample a single JL entry: χ(0) = 1/2, χ(±1) = 1/4 (two random bits).
/// Sparser than the ternary distribution used in rok_fold (which is p = 1/3 each).
/// The JL property only needs mean = 0 and bounded variance, so the sparse
/// distribution is cheaper at no soundness cost.
fn sample_j_entry<const Q: u64, const D: usize>(_rng: &mut impl Rng) -> Rq<Q, D> {
    todo!()
}

/// Sample J ∈ R_q^{n_rp × m_rp} with entries from `sample_j_entry`.
fn sample_j<const Q: u64, const D: usize>(
    n_rp: usize,
    m_rp: usize,
    rng: &mut impl Rng,
) -> Mat<Rq<Q, D>> {
    Mat::<Rq<Q, D>>::from_fn(n_rp, m_rp, |_i, _j| sample_j_entry(rng))
}

/// Column-major flatten of a matrix.
///
/// E.g. W = [[1, 2, 3],
///           [4, 5, 6]]
///      vec(W) = [1, 4, 2, 5, 3, 6]
fn vec_col_major<const Q: u64, const D: usize>(_w: &Mat<Rq<Q, D>>) -> Vec<Rq<Q, D>> {
    todo!()
}

/// Π^⊗RP: prove W satisfies F·W = Y with ‖W‖ ≤ β using Johnson–Lindenstrauss
/// random projection. Returns:
///   - `lin_orig`:  augmented original (H̃, F̃, Ỹ) with W unchanged, plus the
///                  c_1·Ĵ row constraining W via the projection.
///   - `lin_w_hat`: projected ((I, F̂, ŷ), ŵ) — width collapsed to r=1,
///                  m shrinks to m' = m/r, β grows to m_rp · β.
///
/// Precondition: m_rp == n_rp · r and m is divisible by m_rp.
pub fn rok_rp<const Q: u64, const D: usize>(
    lin: &LinRelation<Q, D>,
    n_rp: usize,
    m_rp: usize,
    rng: &mut impl Rng,
) -> (LinRelation<Q, D>, LinRelation<Q, D>) {
    let n_hat = lin.n_hat();
    let m = lin.m();
    let n = lin.n();
    let n_top = lin.n_top();
    let r = lin.r();
    let f_com = &lin.instance.f_com;
    let f_eval = &lin.instance.f_eval;
    let h = &lin.instance.h;
    let y = &lin.instance.y;
    let w = &lin.witness.w;

    assert_eq!(
        m_rp,
        n_rp * r,
        "need m_rp = n_rp·r, got m_rp={m_rp}, n_rp={n_rp}, r={r}",
    );

    //
    // Verifier
    //
    // 1. Sample J and send it to Prover.
    let m_prime = m / r;
    let _j = sample_j::<Q, D>(n_rp, m_rp, rng);

    //
    // Prover
    //
    // 1. Calculate Ĵ = I_{m / m_rp} ⊗ J.
    let _size_i = m / m_rp;
    // I = identity_matrix(Rq, size_i)
    // J_hat = I.tensor_product(J)
    // assert J_hat.nrows() == size_i * n_rp
    // assert J_hat.ncols() == size_i * m_rp

    // 2. Ŵ = Ĵ · W   ∈ R_q^{m' × r}

    // 3. flatten Ŵ to ŵ ∈ R_q^m (column-major)

    // 4. Commit ŵ to save proof size: z̄ = F_com · ŵ
    //    — send `z̄` to Verifier.

    //
    // Verifier
    //
    // 4. Sample c from R_q. Send c to Prover.

    //
    // Prover
    //
    // 5–8. c_0 = c^{m_prime}, c_1 = c
    //      c_0_vec = (c_0^0, c_0^1, ..., c_0^{r-1})
    //      c_1_vec = (c_1^0, c_1^1, ..., c_1^{m_prime - 1})
    //      c_vec   = c_0_vec ⊗ c_1_vec   ∈ R_q^m

    // 9. r_vec = c_1_vec · Ŵ
    //    — send `r_vec` to Verifier.

    //
    // Both Prover and Verifier
    //
    // 10.1. Build the augmented original:
    //   H̃ = [[H,  0],          F̃ = F_com.stack(F_eval.stack(c_1_vec · Ĵ))
    //        [0,  I]],          Ỹ = Y.stack(r_vec)
    //   (lin_orig wraps these with the original W.)

    // 10.2. Build the projected ŵ-side relation:
    //   F̂_eval = [c_vec]                       (1 row, m cols)
    //   ŷ      = column[z̄ ‖ c_0_vec · r_vec]  (n_top + 1 rows, 1 col)
    //   H_hat  = I_{n_top + 1}
    //   β_hat  = m_rp · β
    let _new_beta = (m_rp as u64) * lin.beta();

    let _ = (n_hat, m, n, n_top, r, f_com, f_eval, h, y, w, m_prime);

    todo!()
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
    /// Y = H · F · W. Used here with `f_eval = 0×m` (commitment-only setup,
    /// matching Π^⊗RP's typical input).
    fn build_rel(f_com: Mat<R>, f_eval: Mat<R>, w: Mat<R>, beta: u64) -> LinRelation<Q, D> {
        let n_total = f_com.nrows() + f_eval.nrows();
        let h = Mat::<R>::identity(n_total);
        let f = f_com.stack(&f_eval);
        let y = h.clone() * f * w.clone();
        let inst = LinInstance::new(h, f_com, f_eval, y, beta);
        let wit = LinWitness::new(w);
        LinRelation::new(inst, wit)
    }

    /// Π^⊗RP input: F_com 2 × m, F_eval empty, W m × r with m divisible by r.
    /// Picks m = 4, r = 2 → m' = m/r = 2 (smallest legal: n_rp=1, m_rp=r=2).
    fn make_rp_input() -> LinRelation<Q, D> {
        // F_com: 2 × 4
        let f_com = mat(&[[1, 2, 3, 0], [0, 5, 7, 11]]);
        // F_eval: 0 × 4 (no eval rows)
        let f_eval = Mat::<R>::zero(0, 4);
        // W: 4 × 2, small entries so β = 4 is comfortable.
        let w = mat(&[[1, 0], [0, 1], [1, 0], [0, 1]]);
        build_rel(f_com, f_eval, w, 4)
    }

    // ─── output shape ───

    /// rok_rp returns a 2-tuple of LinRelations: (aug, proj).
    #[test]
    fn test_rp_returns_tuple_of_lin_relations() {
        let lin_in = make_rp_input();
        let n_rp = 1;
        let m_rp = lin_in.r(); // smallest legal: m_rp = n_rp · r = r.
        let mut rng = rand::rng();
        let (aug, proj) = rok_rp(&lin_in, n_rp, m_rp, &mut rng);
        // Reaching here = both tuple components are valid LinRelations.
        let _ = (aug, proj);
    }

    /// Augmented side: appends one row to F_eval and H, leaves W untouched.
    #[test]
    fn test_rp_aug_preserves_dims_except_plus_one_row() {
        let lin_in = make_rp_input();
        let n_rp = 1;
        let m_rp = lin_in.r();
        let mut rng = rand::rng();
        let (aug, _proj) = rok_rp(&lin_in, n_rp, m_rp, &mut rng);

        assert_eq!(
            aug.n_hat(),
            lin_in.n_hat() + 1,
            "n̂ grows by 1 (one extra row)"
        );
        assert_eq!(aug.n(), lin_in.n() + 1, "n grows by 1 (extra column in H)");
        assert_eq!(aug.m(), lin_in.m(), "m unchanged");
        assert_eq!(aug.r(), lin_in.r(), "r unchanged on aug side");
        assert_eq!(aug.beta(), lin_in.beta(), "β unchanged on aug side");
        // Π^⊗RP only appends a row to F_eval; F_com is not touched.
        assert_eq!(aug.instance.f_com, lin_in.instance.f_com, "F_com preserved");
    }

    /// Projected side: r collapses to 1, β grows by m_rp factor.
    #[test]
    fn test_rp_proj_collapses_to_r1_and_grows_beta() {
        let lin_in = make_rp_input();
        let n_rp = 1;
        let m_rp = lin_in.r();
        let mut rng = rand::rng();
        let (_aug, proj) = rok_rp(&lin_in, n_rp, m_rp, &mut rng);

        assert_eq!(proj.r(), 1, "ŵ is a single column");
        assert_eq!(proj.n_hat(), lin_in.n_top() + 1, "n̂ = n_top + 1");
        assert_eq!(proj.n(), lin_in.n_top() + 1, "n = n_top + 1");
        assert_eq!(proj.m(), lin_in.m(), "m unchanged");
        // β̂ = m_rp · β.
        assert_eq!(proj.beta(), (m_rp as u64) * lin_in.beta(), "β̂ = m_rp · β",);
    }

    // ─── relation correctness ───

    /// Both output LinRelations must satisfy H·F·W = Y (LinRelation::new
    /// panics otherwise). Reaching the end without panic confirms the algebra
    /// is internally consistent on both sides.
    #[test]
    fn test_rp_both_produce_valid_relations() {
        let lin_in = make_rp_input();
        let n_rp = 1;
        let m_rp = lin_in.r();
        let mut rng = rand::rng();
        let (_aug, _proj) = rok_rp(&lin_in, n_rp, m_rp, &mut rng);
    }

    // ─── precondition: m_rp = n_rp · r ───

    #[test]
    #[should_panic]
    fn test_rp_m_rp_not_divisible_panics() {
        let lin_in = make_rp_input();
        // r = 2; pick m_rp = 3 (not n_rp · r for any integer n_rp).
        let mut rng = rand::rng();
        let _ = rok_rp(&lin_in, /* n_rp = */ 1, /* m_rp = */ 3, &mut rng);
    }
}
