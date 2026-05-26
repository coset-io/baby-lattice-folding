//! Π^norm: reduces a linear relation to a sum-linear relation via sumcheck
//! over squared coefficients, using the conjugation trick.
//!
//! See `notes-paper/salsaa-norm-check-code-mapping.md` and
//! `06_salsaa/rok/norm.py`.
//!
//! Two sub-protocols:
//!   - `rok_norm`     : prove ‖w_i‖_{a,2} ≤ ν via t_i = ⟨w_i, w̄_i⟩
//!   - `rok_bar_sum`  : sumcheck on RLC of CRT(LDE[W] · LDE[W̄]),
//!                      output verified via s_0 = LDE[W](r̃), s_1 = LDE[W](r̃̄)
//!
//! BLOCKED ON:
//!   - LDE module (`src/lde.rs` — tensor_product, lde_poly). Not yet built.
//!   - `Rq::conjugate` (Galois map X ↦ X^{-1} on R_q). Not yet built.
//!   - `sumcheck::sumcheck` (just-added skeleton, also a stub).
//! These dependencies are flagged where used; the stubs below give the shape
//! the SALSAA Python prototype expects so the translation has clear anchor points.

use rand::Rng;

use crate::{lde::{lde, tensor}, mat::Mat, relations::{LinInstance, LinRelation, LinWitness}, ring::Rq, zq::Zq};

/// Sample u ∈ Z_q\{0} and return the Vandermonde column (u^0, u^1, ..., u^{r·d/e - 1}).
///
/// Used by `rok_bar_sum` as the RLC coefficient vector across all NTT slots
/// (there are r·d/e NTT slots total: r columns × d/e slots per Rq element).
pub fn sample_u_vec<const Q: u64, const D: usize>(
    r: usize,
    e: usize,
    rng: &mut impl Rng,
) -> Mat<Zq<Q>> {
    // u = random nonzero Z_q
    let mut u = Zq::<Q>::random(rng);
    while u == Zq::zero() {
        u = Zq::<Q>::random(rng);
    }
    // return [u^0, u^1, ..., u^{r·d/e - 1}]
    Mat::from_fn(1, r * D / e, |i, j| u.pow((i * j) as u64))
}

fn get_l(m: usize, d_h: usize) -> usize {
    let mut l = 1;
    let mut cur_hc_size = d_h;
    while cur_hc_size < m {
        cur_hc_size *= d_h;
        l += 1;
    }
    l
}

/// Π^bar-sum: sumcheck on the RLC of CRT(LDE[W] · LDE[W̄]).
///
/// Returns `((r_T, s_0), (r_T_bar, s_1))` where:
///   - `r_T`     ∈ R_q^l: the (lifted-to-Rq) sumcheck challenge vector.
///   - `r_T_bar` ∈ R_q^l: `r_T` under conjugation (Galois map).
///   - `s_0`     ∈ R_q^r: LDE[W] evaluated at r_T (per-column).
///   - `s_1`     ∈ R_q^r: LDE[W] evaluated at r_T_bar (per-column).
///
/// Verifier checks `a_l ?= u^T · CRT(s_0 · s̄_1)` to close the loop.
#[allow(clippy::type_complexity)]
pub fn rok_bar_sum<const Q: u64, const D: usize>(
    e: usize,
    d_h: usize,
    t_vec: &[Rq<Q, D>],
    w_mat: &Mat<Rq<Q, D>>,
    rng: &mut impl Rng,
) -> (
    (Vec<Rq<Q, D>>, Vec<Rq<Q, D>>),
    (Vec<Rq<Q, D>>, Vec<Rq<Q, D>>),
) {
    let m = w_mat.nrows();
    let r = w_mat.ncols();
    assert_eq!(t_vec.len(), r);

    //
    // Verifier
    //

    // Challenges in RLC for all NTT slots
    // We have a LDE for each column w_i \in W=[w_1, ..., w_r] and
    // we split that LDE into d/e NTT slots f_\text{slot_0}, ..., f_\text{slot_3}
    // So in total there are r*d/e slots (F_{q^e})
    // u^T: [u^0, u^1, ..., u^{rd/e}]
    let u_vec = sample_u_vec::<Q,D>(r, e, rng);

    // [t_{0,0}, ..., t_{0,d/e}, ..., t_{r-1,0}, ..., t_{r-1,d/e}]
    // t_ntt = [
    //     t_i_s
    //     for t_i in t  # t_i \in R_q
    //     for t_i_s in ntt(t_i.list(), Rq)
    // ]

    // t_ntt = flatten([NTT(t_i) for t_i in t])   ∈ Z_q^{r·d/e}
    //         = [t_{0,0}, ..., t_{0, d/e-1}, ..., t_{r-1, 0}, ..., t_{r-1, d/e-1}]
    //
    // a_0 = u_T · t_ntt   (initial sumcheck claim)

    //
    // Prover
    //
    // For each column w_i:
    //   w_i_bar = conjugate(w_i)
    //   lde_w   = LDE[w_i],  lde_w_bar = LDE[w_i_bar]
    //   prod    = lde_w * lde_w_bar     ∈ R_q[x_0, ..., x_{l-1}]
    //
    //   Decompose `prod` into d/e NTT slots:
    //     For each monomial m with coefficient c (an R_q element):
    //       c_ntt = NTT(c)              ∈ Z_q^d   (d/e slots × e per slot)
    //     tilde_f_slot_i  =  Σ_monomials  c_ntt[i] · m       (multivariate over Z_q)
    //   crt_LDE_W_LDE_bar_W.extend([tilde_f_slot_0, ..., tilde_f_slot_{d/e - 1}])
    //   lde_W.append(lde_w)
    //
    // tilde_f = Σ_i u_T[i] · crt_LDE_W_LDE_bar_W[i]   (single multivariate poly over Z_q)

    //
    // Prover ↔ Verifier: sumcheck on tilde_f
    //
    // a_l, rands = sumcheck(tilde_f, xs, a_0, D_hypercube)
    let l = get_l(m, d_h);
    
    // FIXME: now it's calculated directly from prover.
    let a_l: Zq<Q> = Zq::zero();
    let rs: Vec<Zq<Q>> = (0..l).map(|_| Zq::new(rng.random_range(0..Q))).collect();

    //
    // Prover: oracle-check side
    //
    // For each sumcheck challenge r_j (a Z_q scalar), lift to R_q by inverse-NTT
    // of the constant vector (r_j repeated d/e times):
    //   r_T[j] = iNTT([r_j; d/e])    ∈ R_q
    //
    // s_0 = [LDE[w_i].subs(r_T) for w_i in W.columns]    ∈ R_q^r
    // s_1 = [LDE[w_i].subs(r_T_bar) for w_i in W.columns] ∈ R_q^r
    //   where r_T_bar = [conjugate(r_T[j]) for j in 0..l].
    //
    // Send s_0, s_1 to Verifier.
    let r_vec: Vec<Rq<Q, D>> = rs.iter().map(|&r_f| Rq::<Q,D>::from_zq(r_f)).collect();
    let s_0: Vec<_> = (0..r).map(
        |j| {
            let w_j = w_mat.col(j);
            lde(&w_j, d_h, &r_vec)
        }
    ).collect();
    let r_bar_vec: Vec<_> = r_vec.iter().map(|r_r| r_r.conjugate()).collect();
    let s_1: Vec<_> = (0..r).map(
        |j| {
            let w_j = w_mat.col(j);
            lde(&w_j, d_h, &r_vec)
        }
    ).collect();

    //
    // Verifier (final check)
    //
    // For each i: s0_s1_bar = s_0[i] · conjugate(s_1[i])
    // rhs_ntt   = CRT(s0_s1_bar) = flatten([NTT(s0_s1_bar[i]) for i in 0..r])
    // rhs       = u_T · rhs_ntt
    // assert a_l == rhs

    (
        (r_vec, s_0),
        (r_bar_vec, s_1),
    )
}

/// Π^norm: prove ‖W‖₂² ≤ d · β² by reducing to a sum-linear relation via
/// `rok_bar_sum` (sumcheck on CRT(LDE[W]·LDE[W̄])), then embed the two
/// evaluation points (r_0, r_1) into F_eval and (s_0, s_1) into Y.
///
/// Effect: n̂ += 2, n += 2; m, r, β preserved. F_com untouched.
pub fn rok_norm<const Q: u64, const D: usize>(e: usize, d_h: usize, rng: &mut impl Rng, lin: &LinRelation<Q, D>) -> LinRelation<Q, D> {
    let h  = &lin.instance.h;
    let f_com = &lin.instance.f_com;
    let f_eval = &lin.instance.f_eval;
    let y = &lin.instance.y;
    let w = &lin.witness.w;
    let m = lin.m();
    let r = lin.r();

    //
    // Prover
    //
    // t_i = Σ_{j=0..m} w_i[j] · conjugate(w_i[j])    for i in 0..r
    // (i.e. t_i = ⟨w_i, w̄_i⟩  ∈ R_q)
    // Send `t` to Verifier.

    let t_vec: Vec<_> = (0..r).map(|j| {
        let col = w.col(j);
        let t_i = col.iter()
            .map(|w_ij| w_ij.clone() * w_ij.conjugate())
            .fold(Rq::zero(), |acc, v| acc + v);
        t_i
    }).collect();

    //
    // Verifier: bound check
    //
    // μ² := d · β²
    // For each i: Trace(t_i) = d · constant_term(t_i)
    //   assert Trace(t_i) ≤ μ²
    let mu_square = (D as u64) * (lin.beta() * lin.beta());
    for i in 0..r {
        // Trace(t_i) = d * ct(t_i)
        let trace_t_i = (D as u64) * t_vec[i].coeffs()[0].value();
        assert!(trace_t_i < mu_square);
    }

    //
    // Prover ↔ Verifier: reduce  t_i ?= ⟨w_i, w̄_i⟩  to  "LDE[W](r_0) = s_0, LDE[W](r_1) = s_1"
    //
    // ((r_0, s_0), (r_1, s_1)) = rok_bar_sum(r, t, W)
    let ((r_0, s_0), (r_1, s_1)) = rok_bar_sum(e, d_h, &t_vec, w, rng);

    //
    // Both: embed (s_0, s_1) into the existing relation H·F·W = Y
    //
    // new_F_rows = [tensor_product(r_0, D_hypercube),
    //               tensor_product(r_1, D_hypercube)]

    let new_f_rows = Mat::<Rq<Q,D>>::new(vec![
        tensor(&r_0, d_h),
        tensor(&r_1, d_h)
    ]);
    // new_Y_rows = [s_0, s_1]

    let new_y_rows = Mat::<Rq<Q,D>>::new(vec![
        s_0,
        s_1,
    ]);
    // lin_new = LinRelation(
    //     instance = lin.instance.with_extra_eval(new_F_rows, new_Y_rows),
    //     witness  = lin.witness,
    // )
    // Result: n̂ += 2, n += 2, m unchanged, r unchanged, β unchanged.

    let _ = (w, m, r);
    let f_eval_new = f_eval.stack(&new_f_rows);
    let y_new = y.stack(&new_y_rows);
    // H grows to match: 2 new eval rows ⇒ append I_2 along the diagonal so
    // each new constraint is an identity selector against the new F_eval rows.
    let h_new = Mat::block_diagonal(&[h.clone(), Mat::identity(2)]);


    // H * F * \tilde W = \tilde Y
    LinRelation::new(
        LinInstance::new(h_new, f_com.clone(), f_eval_new, y_new, lin.beta()),
        LinWitness::new(w.clone()),
    )
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

    /// Π^norm appends two F_eval rows and two Y rows; n̂ and n both grow by 2.
    /// m, r, β preserved. F_com untouched.
    #[test]
    fn test_norm_appends_two_rows_to_eval_and_y() {
        let rel = build_rel(
            mat(&[[1, 2]]),       // F_com: 1 × 2 → n_top = 1
            Mat::<R>::zero(0, 2), // F_eval: empty (norm runs at start of chain)
            mat(&[[1], [0]]),     // W: 2 × 1, small norm
            4,
        );
        let mut rng = rand::rng();
        let out = rok_norm(/* e = */ 1, /* d_h = */ 2, &mut rng, &rel);

        assert_eq!(out.n_hat(), rel.n_hat() + 2, "n̂ += 2");
        assert_eq!(out.n(), rel.n() + 2, "n += 2");
        assert_eq!(out.m(), rel.m(), "m unchanged");
        assert_eq!(out.r(), rel.r(), "r unchanged");
        assert_eq!(out.beta(), rel.beta(), "β unchanged");
        assert_eq!(out.n_top(), rel.n_top(), "n_top unchanged");
        assert_eq!(out.instance.f_com, rel.instance.f_com, "F_com preserved");
    }

    /// rok_norm MUST produce a `LinRelation` whose `H · F · W = Y` holds on
    /// the extended instance. Reaching the end without panic = LinRelation::new
    /// validated the extended algebra.
    #[test]
    fn test_norm_produces_valid_relation() {
        let rel = build_rel(mat(&[[1, 2]]), Mat::<R>::zero(0, 2), mat(&[[1], [0]]), 4);
        let mut rng = rand::rng();
        let _out = rok_norm(/* e = */ 1, /* d_h = */ 2, &mut rng, &rel);
    }

    // Norm-bound violation isn't tested here on purpose: LinRelation::new
    // already enforces ‖w_i‖₂ ≤ β at construction time, so a violating witness
    // can never reach rok_norm. The bound Π^norm itself checks is the tighter
    // ‖w_i‖_{a,2} ≤ ν (see paper §3 / salsaa-norm-check-code-mapping.md), and
    // a meaningful test for that needs the implementation to settle first.
}
