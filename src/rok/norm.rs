use rand::Rng;

use crate::{
    lde::{lde, tensor},
    mat::Mat,
    relations::{LinInstance, LinRelation, LinWitness},
    ring::Rq,
    sumcheck::{SumcheckProver, fold_table, h_x, sumcheck},
    zq::Zq,
};

fn sample_u_vec<const Q: u64, const D: usize>(
    r: usize,
    e: usize,
    rng: &mut impl Rng,
) -> Vec<Zq<Q>> {
    // u = random nonzero Z_q
    let mut u = Zq::<Q>::random(rng);
    while u == Zq::zero() {
        u = Zq::<Q>::random(rng);
    }
    // return [u^0, u^1, ..., u^{r·d/e - 1}]
    let n = r * D / e;
    (0..n).map(|j| u.pow(j as u64)).collect()
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

/// Pad `w` along rows with zero polynomials so its row count is `d^l`, where
/// `l = ⌈log_d(m)⌉` (smallest hypercube `[d]^l` containing m rows). Returns
/// `(padded_w, l)`.
fn pad_mat_to_d_exp<const Q: u64, const D: usize>(
    w: &Mat<Rq<Q, D>>,
    d: usize,
) -> (Mat<Rq<Q, D>>, usize) {
    let m = w.nrows();
    let l = get_l(m, d);
    let target = d.pow(l as u32);
    if target == m {
        return (w.clone(), l);
    }
    let pad = Mat::<Rq<Q, D>>::zero(target - m, w.ncols());
    (w.stack(&pad), l)
}

pub struct BatchedSlotProver<const Q: u64, const D: usize> {
    // Each has r*D/e slots (r columns, and D/e per LDE)
    table_w: Vec<Vec<Zq<Q>>>,
    table_w_bar: Vec<Vec<Zq<Q>>>,
    u_vec: Vec<Zq<Q>>,
}

impl<const Q: u64, const D: usize> BatchedSlotProver<Q, D> {
    pub fn new(w: &Mat<Rq<Q, D>>, u_vec: Vec<Zq<Q>>, e: usize) -> Self {
        assert_eq!(e, 1, "e > 1 not supported yet");
        let r = w.ncols();
        let expected_vec_len = r * D / e;
        assert_eq!(u_vec.len(), expected_vec_len);
        let mut table_w: Vec<Vec<Zq<Q>>> = vec![Vec::new(); expected_vec_len];
        let mut table_w_bar: Vec<Vec<Zq<Q>>> = vec![Vec::new(); expected_vec_len];
        for j in 0..r {
            let col = w.col(j);
            let mut table_w_col_j: Vec<Vec<Zq<Q>>> = vec![Vec::new(); D / e];
            let mut table_w_bar_col_j: Vec<Vec<Zq<Q>>> = vec![Vec::new(); D / e];
            for w_ij in col.iter() {
                let w_j_slots = w_ij.ntt();
                let w_j_bar = w_ij.conjugate();
                let w_j_bar_slots = w_j_bar.ntt();
                for (k, (&w_slot_k, &w_bar_slot_k)) in w_j_slots
                    .evals()
                    .iter()
                    .zip(w_j_bar_slots.evals())
                    .enumerate()
                {
                    table_w_col_j[k].push(w_slot_k);
                    table_w_bar_col_j[k].push(w_bar_slot_k);
                }
            }
            for k in 0..D / e {
                table_w[j * (D / e) + k] = table_w_col_j[k].clone();
                table_w_bar[j * (D / e) + k] = table_w_bar_col_j[k].clone();
            }
        }
        Self {
            table_w,
            table_w_bar,
            u_vec,
        }
    }
}

impl<const Q: u64, const D: usize> SumcheckProver<Q> for BatchedSlotProver<Q, D> {
    fn round_message(&self) -> Vec<Zq<Q>> {
        // LDE[W] \cdot LDE[\bar W] is degree-2 per variable → 3 evaluations per round.
        const TARGET_DEGREE: usize = 2;
        (0..=TARGET_DEGREE as u64)
            .map(|x| {
                (0..self.u_vec.len())
                    .map(|i| {
                        self.u_vec[i] * h_x(&self.table_w[i], &self.table_w_bar[i], Zq::new(x))
                    })
                    .fold(Zq::zero(), |acc, v| acc + v)
            })
            .collect()
    }

    fn fold(&mut self, r: Zq<Q>) {
        for k in 0..self.table_w.len() {
            self.table_w[k] = fold_table(&self.table_w[k], r);
            self.table_w_bar[k] = fold_table(&self.table_w_bar[k], r);
        }
    }

    fn final_value(&self) -> Zq<Q> {
        // After all rounds, each table is length 1 = MLE evaluated at rands.
        let mut total = Zq::<Q>::zero();
        // Go through each slot and sum w * w_bar
        for k in 0..self.table_w.len() {
            assert_eq!(self.table_w[k].len(), 1);
            assert_eq!(self.table_w_bar[k].len(), 1);
            total = total + self.u_vec[k] * self.table_w[k][0] * self.table_w_bar[k][0]
        }
        total
    }
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
    w: &Mat<Rq<Q, D>>,
    rng: &mut impl Rng,
) -> (
    (Vec<Rq<Q, D>>, Vec<Rq<Q, D>>),
    (Vec<Rq<Q, D>>, Vec<Rq<Q, D>>),
) {
    let r = w.ncols();
    assert_eq!(t_vec.len(), r);
    assert_eq!(e, 1, "rok_bar_sum: e > 1 not supported yet");

    // Pad W to the [d_h]^l hypercube
    let (w_padded, l) = pad_mat_to_d_exp(w, d_h);

    //
    // Verifier
    //
    // Challenges in RLC for all NTT slots
    // We have a LDE for each column w_i \in W=[w_1, ..., w_r] and
    // we split that LDE into d/e NTT slots f_\text{slot_0}, ..., f_\text{slot_3}
    // So in total there are r*d/e slots (F_{q^e})
    // u^T: [u^0, u^1, ..., u^{rd/e}]
    let u_vec = sample_u_vec::<Q, D>(r, e, rng);
    let t_ntt: Vec<Zq<Q>> = t_vec
        .iter()
        .flat_map(|t_i| t_i.ntt().evals().to_vec())
        .collect();
    assert_eq!(u_vec.len(), t_ntt.len());
    let a_0 = u_vec
        .iter()
        .zip(t_ntt)
        .map(|(&u_i, t_i)| u_i * t_i)
        .fold(Zq::zero(), |acc, v| acc + v);

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
    let mut batch_prover = BatchedSlotProver::<Q, D>::new(&w_padded, u_vec.clone(), e);

    //
    // Prover ↔ Verifier: sumcheck on tilde_f
    //
    let out = sumcheck(&mut batch_prover, a_0, l, d_h, rng);
    let a_l: Zq<Q> = out.a_l;
    let rs: Vec<Zq<Q>> = out.rands;

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
    let r_vec: Vec<Rq<Q, D>> = rs.iter().map(|&r_f| Rq::<Q, D>::from_zq(r_f)).collect();
    let s_0_vec: Vec<_> = (0..r)
        .map(|j| {
            let w_j = w_padded.col(j);
            lde(&w_j, d_h, &r_vec)
        })
        .collect();
    let r_bar_vec: Vec<_> = r_vec.iter().map(|r_r| r_r.conjugate()).collect();
    let s_1_vec: Vec<_> = (0..r)
        .map(|j| {
            let w_j = w_padded.col(j);
            lde(&w_j, d_h, &r_bar_vec)
        })
        .collect();
    // Send s_0, s_1 to Verifier.

    //
    // Verifier
    //
    // For each i: s0_s1_bar = s_0[i] · conjugate(s_1[i])
    // rhs_ntt   = CRT(s0_s1_bar) = flatten([NTT(s0_s1_bar[i]) for i in 0..r])
    // rhs       = u_T · rhs_ntt
    let s_0_s_1_bar_vec: Vec<Zq<Q>> = s_0_vec
        .iter()
        .zip(&s_1_vec)
        .flat_map(|(&s_0, s_1)| {
            let t = s_0 * s_1.conjugate();
            t.ntt().evals().to_vec()
        })
        .collect();
    let rhs = u_vec
        .iter()
        .zip(s_0_s_1_bar_vec)
        .map(|(&u, s_0_s_1)| u * s_0_s_1)
        .fold(Zq::<Q>::zero(), |acc, v| acc + v);
    assert_eq!(a_l, rhs);

    ((r_vec, s_0_vec), (r_bar_vec, s_1_vec))
}

/// Π^norm: prove ‖W‖₂² ≤ d · β² by reducing to a sum-linear relation via
/// `rok_bar_sum` (sumcheck on CRT(LDE[W]·LDE[W̄])), then embed the two
/// evaluation points (r_0, r_1) into F_eval and (s_0, s_1) into Y.
///
/// Effect: n̂ += 2, n += 2; m, r, β preserved. F_com untouched.
pub fn rok_norm<const Q: u64, const D: usize>(
    e: usize,
    d_h: usize,
    rng: &mut impl Rng,
    lin: &LinRelation<Q, D>,
) -> LinRelation<Q, D> {
    let h = &lin.instance.h;
    let f_com = &lin.instance.f_com;
    let f_eval = &lin.instance.f_eval;
    let y = &lin.instance.y;
    let w = &lin.witness.w;
    let r = lin.r();

    //
    // Prover
    //
    // t_i = Σ_{j=0..m} w_i[j] · conjugate(w_i[j])    for i in 0..r
    // (i.e. t_i = ⟨w_i, w̄_i⟩  ∈ R_q)
    // Send `t` to Verifier.
    let t_vec: Vec<_> = (0..r)
        .map(|j| {
            let col = w.col(j);

            col.iter()
                .map(|w_ij| *w_ij * w_ij.conjugate())
                .fold(Rq::zero(), |acc, v| acc + v)
        })
        .collect();

    //
    // Verifier: bound check
    //
    // μ² := d · β²
    // For each i: Trace(t_i) = d · constant_term(t_i)
    //   assert Trace(t_i) ≤ μ²
    let mu_square = (D as u64) * (lin.beta() * lin.beta());
    for t_i in &t_vec {
        // Trace(t_i) = d * ct(t_i)
        let trace_t_i = (D as u64) * t_i.coeffs()[0].value();
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
    let new_f_rows = Mat::<Rq<Q, D>>::new(vec![tensor(&r_0, d_h), tensor(&r_1, d_h)]);

    // new_Y_rows = [s_0, s_1]
    let new_y_rows = Mat::<Rq<Q, D>>::new(vec![s_0, s_1]);

    // Check relation holds
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
    use crate::sumcheck::sumcheck;
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

    // ─── BatchedSlotProver ───

    /// Build an Rq element from per-coefficient u64s (allows non-constant polys
    /// so NTT slots are non-trivial, unlike `c(v)` which produces all-equal slots).
    fn rq(coeffs: [u64; D]) -> R {
        R::new(coeffs.map(Zq::<Q>::new))
    }

    /// u_vec = [u^0, u^1, ..., u^{r·D/e - 1}] for a random nonzero u.
    fn make_u_vec(r: usize, e: usize, rng: &mut impl Rng) -> Vec<Zq<Q>> {
        let mut u = Zq::<Q>::random(rng);
        while u == Zq::zero() {
            u = Zq::<Q>::random(rng);
        }
        (0..r * D / e).map(|i| u.pow(i as u64)).collect()
    }

    /// Compute a_0 = u^T · t_ntt the way `rok_bar_sum` will: per column
    /// t_j = ⟨w_j, w̄_j⟩ ∈ R_q, NTT-decompose to slots, dot with u_vec.
    ///
    /// Relies on NTT being a ring hom: NTT(a·b) = NTT(a)·NTT(b) (pointwise) and
    /// NTT(Σ ...) = Σ NTT(...). So NTT(t_j)[k] = Σ_x NTT(w[x,j])[k]·NTT(w̄[x,j])[k]
    /// which is exactly the per-slot inner product the prover sums internally.
    fn compute_claimed(w: &Mat<R>, u_vec: &[Zq<Q>], e: usize) -> Zq<Q> {
        let mut total = Zq::<Q>::zero();
        for j in 0..w.ncols() {
            let col = w.col(j);
            let t_j = col
                .iter()
                .map(|w_ij| *w_ij * w_ij.conjugate())
                .fold(R::zero(), |acc, v| acc + v);
            let t_j_ntt = t_j.ntt();
            for k in 0..(D / e) {
                total = total + u_vec[j * (D / e) + k] * t_j_ntt.evals()[k];
            }
        }
        total
    }

    /// Rebuild per-slot tables OUTSIDE the prover, fold via `fold_table`
    /// directly, then RLC. Independent of BatchedSlotProver::{new, fold,
    /// final_value} — catches coordinated bugs in all three that an end-to-end
    /// round-trip alone could miss.
    fn independent_expected_a_l(w: &Mat<R>, u_vec: &[Zq<Q>], e: usize, rands: &[Zq<Q>]) -> Zq<Q> {
        let r_cols = w.ncols();
        let mut tw: Vec<Vec<Zq<Q>>> = Vec::with_capacity(r_cols * D / e);
        let mut tw_bar: Vec<Vec<Zq<Q>>> = Vec::with_capacity(r_cols * D / e);
        for j in 0..r_cols {
            let col = w.col(j);
            let mut per_col_w: Vec<Vec<Zq<Q>>> = vec![Vec::new(); D / e];
            let mut per_col_w_bar: Vec<Vec<Zq<Q>>> = vec![Vec::new(); D / e];
            for w_ij in col.iter() {
                let slots = w_ij.ntt();
                let slots_bar = w_ij.conjugate().ntt();
                for k in 0..(D / e) {
                    per_col_w[k].push(slots.evals()[k]);
                    per_col_w_bar[k].push(slots_bar.evals()[k]);
                }
            }
            for k in 0..(D / e) {
                tw.push(per_col_w[k].clone());
                tw_bar.push(per_col_w_bar[k].clone());
            }
        }
        for &r in rands {
            for t in tw.iter_mut() {
                *t = fold_table(t, r);
            }
            for t in tw_bar.iter_mut() {
                *t = fold_table(t, r);
            }
        }
        (0..tw.len())
            .map(|i| u_vec[i] * tw[i][0] * tw_bar[i][0])
            .fold(Zq::zero(), |acc, v| acc + v)
    }

    /// End-to-end round-trip on r=2, m=2 (one sumcheck round).
    ///
    /// Asserts: (1) sumcheck completes without round-msg mismatch
    ///          (2) driver's a_l matches independent table-fold + RLC
    ///          (3) prover.final_value() matches a_l
    ///
    /// (1) covers `round_message` (and indirectly `new`'s slot layout — wrong
    /// u^i ↔ slot pairing would make g(0)+g(1) ≠ claimed). (2) is the deep
    /// cross-check on `new` + `fold` + `final_value` against a path that
    /// doesn't go through ANY of those methods.
    #[test]
    fn test_batched_slot_prover_end_to_end_l1() {
        let e = 1;
        // Non-constant Rq entries → non-trivial NTT slots → real test of slot layout.
        let w = Mat::new(vec![
            vec![rq([1, 2, 0, 3]), rq([5, 0, 1, 0])],
            vec![rq([0, 4, 1, 2]), rq([3, 1, 0, 6])],
        ]);
        let mut rng = rand::rng();
        let u_vec = make_u_vec(w.ncols(), e, &mut rng);
        let claimed = compute_claimed(&w, &u_vec, e);

        let mut prover = BatchedSlotProver::<Q, D>::new(&w, u_vec.clone(), e);
        let out = sumcheck(
            &mut prover,
            claimed,
            /* num_vars = */ 1,
            /* d_h = */ 2,
            &mut rng,
        );
        assert_eq!(out.rands.len(), 1);

        let expected = independent_expected_a_l(&w, &u_vec, e, &out.rands);
        assert_eq!(out.a_l, expected, "driver a_l should match independent RLC");
        assert_eq!(
            prover.final_value(),
            expected,
            "prover.final_value() should match independent RLC"
        );
    }

    /// l=2 (m=4, r=1) — two sumcheck rounds, catches off-by-one in fold loops
    /// and any "first-round special case" assumption that breaks on round 2.
    #[test]
    fn test_batched_slot_prover_end_to_end_l2() {
        let e = 1;
        let w = Mat::new(vec![
            vec![rq([1, 2, 0, 3])],
            vec![rq([0, 4, 1, 2])],
            vec![rq([5, 0, 1, 0])],
            vec![rq([3, 1, 0, 6])],
        ]);
        let mut rng = rand::rng();
        let u_vec = make_u_vec(w.ncols(), e, &mut rng);
        let claimed = compute_claimed(&w, &u_vec, e);

        let mut prover = BatchedSlotProver::<Q, D>::new(&w, u_vec.clone(), e);
        let out = sumcheck(
            &mut prover,
            claimed,
            /* num_vars = */ 2,
            /* d_h = */ 2,
            &mut rng,
        );
        assert_eq!(out.rands.len(), 2);

        let expected = independent_expected_a_l(&w, &u_vec, e, &out.rands);
        assert_eq!(out.a_l, expected);
        assert_eq!(prover.final_value(), expected);
    }

    /// Wrong claim must trip the round-0 verifier check inside the driver.
    /// Confirms the BatchedSlotProver's round_message agrees with the claim
    /// computation path used by `compute_claimed` (i.e. the protocol contract).
    #[test]
    #[should_panic(expected = "round-message claim mismatch")]
    fn test_batched_slot_prover_wrong_claim_panics() {
        let e = 1;
        let w = Mat::new(vec![
            vec![rq([1, 2, 0, 3]), rq([5, 0, 1, 0])],
            vec![rq([0, 4, 1, 2]), rq([3, 1, 0, 6])],
        ]);
        let mut rng = rand::rng();
        let u_vec = make_u_vec(w.ncols(), e, &mut rng);
        let real = compute_claimed(&w, &u_vec, e);
        // Perturb the claim — any wrong value triggers the round-0 check.
        let bogus = real + Zq::<Q>::one();
        let mut prover = BatchedSlotProver::<Q, D>::new(&w, u_vec, e);
        let _ = sumcheck(&mut prover, bogus, 1, 2, &mut rng);
    }

    /// Constructor asserts u_vec length = r · D/e.
    #[test]
    #[should_panic]
    fn test_batched_slot_prover_wrong_u_vec_len_panics() {
        let w = mat(&[[1, 2], [3, 4]]); // r = 2, D/e = 4 → expects 8
        let bad_u_vec = vec![Zq::<Q>::one(); 3];
        let _ = BatchedSlotProver::<Q, D>::new(&w, bad_u_vec, 1);
    }

    /// r=3 (uneven slot indexing) — catches off-by-one in j*(D/e)+k mapping
    /// that the 2-column tests wouldn't expose if D/e happened to coincide
    /// with r in some symmetric way.
    #[test]
    fn test_batched_slot_prover_r3() {
        let e = 1;
        let w = Mat::new(vec![
            vec![rq([1, 2, 0, 3]), rq([5, 0, 1, 0]), rq([0, 1, 0, 2])],
            vec![rq([0, 4, 1, 2]), rq([3, 1, 0, 6]), rq([2, 0, 3, 1])],
        ]);
        let mut rng = rand::rng();
        let u_vec = make_u_vec(w.ncols(), e, &mut rng);
        let claimed = compute_claimed(&w, &u_vec, e);

        let mut prover = BatchedSlotProver::<Q, D>::new(&w, u_vec.clone(), e);
        let out = sumcheck(&mut prover, claimed, 1, 2, &mut rng);
        let expected = independent_expected_a_l(&w, &u_vec, e, &out.rands);
        assert_eq!(out.a_l, expected);
        assert_eq!(prover.final_value(), expected);
    }

    /// l=3 (m=8, r=1) — three fold rounds, deeper stress on fold loop.
    #[test]
    fn test_batched_slot_prover_l3() {
        let e = 1;
        let w = Mat::new(vec![
            vec![rq([1, 2, 0, 3])],
            vec![rq([0, 4, 1, 2])],
            vec![rq([5, 0, 1, 0])],
            vec![rq([3, 1, 0, 6])],
            vec![rq([2, 2, 0, 0])],
            vec![rq([1, 0, 3, 4])],
            vec![rq([0, 1, 2, 1])],
            vec![rq([4, 3, 1, 0])],
        ]);
        let mut rng = rand::rng();
        let u_vec = make_u_vec(w.ncols(), e, &mut rng);
        let claimed = compute_claimed(&w, &u_vec, e);

        let mut prover = BatchedSlotProver::<Q, D>::new(&w, u_vec.clone(), e);
        let out = sumcheck(&mut prover, claimed, 3, 2, &mut rng);
        assert_eq!(out.rands.len(), 3);
        let expected = independent_expected_a_l(&w, &u_vec, e, &out.rands);
        assert_eq!(out.a_l, expected);
        assert_eq!(prover.final_value(), expected);
    }

    /// BatchedSlotProver::new with e > 1 must panic (assert at top of `new`).
    #[test]
    #[should_panic(expected = "e > 1 not supported yet")]
    fn test_batched_slot_prover_e_gt_1_panics() {
        let w = mat(&[[1, 2], [3, 4]]);
        // Any u_vec length works — the e assert fires first.
        let u_vec = vec![Zq::<Q>::one(); 4];
        let _ = BatchedSlotProver::<Q, D>::new(&w, u_vec, /* e = */ 2);
    }

    // ─── rok_bar_sum ───

    /// Compute t_vec = [⟨w_j, w̄_j⟩ for j in 0..r] for a given W.
    /// rok_bar_sum trusts the caller to supply this consistently with W; the
    /// internal `assert_eq!(a_l, rhs)` will fail otherwise.
    fn compute_t_vec(w: &Mat<R>) -> Vec<R> {
        (0..w.ncols())
            .map(|j| {
                w.col(j)
                    .iter()
                    .map(|w_ij| *w_ij * w_ij.conjugate())
                    .fold(R::zero(), |acc, v| acc + v)
            })
            .collect()
    }

    /// End-to-end: rok_bar_sum returns shapes consistent with the spec and
    /// the internal `assert_eq!(a_l, rhs)` passes (i.e. soundness loop closes).
    #[test]
    fn test_rok_bar_sum_smoke_l1() {
        let w = Mat::new(vec![
            vec![rq([1, 2, 0, 3]), rq([5, 0, 1, 0])],
            vec![rq([0, 4, 1, 2]), rq([3, 1, 0, 6])],
        ]);
        let t_vec = compute_t_vec(&w);
        let mut rng = rand::rng();
        let ((r_t, s_0), (r_t_bar, s_1)) =
            rok_bar_sum::<Q, D>(/* e = */ 1, /* d_h = */ 2, &t_vec, &w, &mut rng);
        assert_eq!(r_t.len(), 1, "l = log_2(m=2) = 1");
        assert_eq!(r_t_bar.len(), r_t.len());
        assert_eq!(s_0.len(), w.ncols(), "s_0 has r entries");
        assert_eq!(s_1.len(), w.ncols(), "s_1 has r entries");
        // r_t_bar entries should be the conjugate of r_t entries.
        for (r_i, r_bar_i) in r_t.iter().zip(&r_t_bar) {
            assert_eq!(*r_bar_i, r_i.conjugate(), "r_bar[i] should be conj(r[i])");
        }
    }

    /// l=2 + r=3: deeper sumcheck with non-trivial column count.
    #[test]
    fn test_rok_bar_sum_smoke_l2_r3() {
        let w = Mat::new(vec![
            vec![rq([1, 2, 0, 3]), rq([5, 0, 1, 0]), rq([0, 1, 0, 2])],
            vec![rq([0, 4, 1, 2]), rq([3, 1, 0, 6]), rq([2, 0, 3, 1])],
            vec![rq([2, 2, 0, 0]), rq([1, 0, 3, 4]), rq([0, 1, 2, 1])],
            vec![rq([4, 3, 1, 0]), rq([0, 2, 1, 3]), rq([3, 0, 0, 1])],
        ]);
        let t_vec = compute_t_vec(&w);
        let mut rng = rand::rng();
        let ((r_t, s_0), (_, s_1)) = rok_bar_sum::<Q, D>(1, 2, &t_vec, &w, &mut rng);
        assert_eq!(r_t.len(), 2, "l = log_2(m=4) = 2");
        assert_eq!(s_0.len(), 3);
        assert_eq!(s_1.len(), 3);
    }

    /// e > 1 must panic at the protocol-layer assert (before BatchedSlotProver
    /// would also panic on the same condition).
    #[test]
    #[should_panic(expected = "e > 1 not supported yet")]
    fn test_rok_bar_sum_e_gt_1_panics() {
        let w = mat(&[[1, 2], [3, 4]]);
        let t_vec = compute_t_vec(&w);
        let mut rng = rand::rng();
        let _ = rok_bar_sum::<Q, D>(/* e = */ 2, 2, &t_vec, &w, &mut rng);
    }

    /// m=3 → pads to d_h^l = 4, sumcheck runs over the padded hypercube, and
    /// the internal verifier check `assert_eq!(a_l, rhs)` passes. This is the
    /// regression guard for `pad_mat_to_d_exp` and for prover/verifier
    /// agreeing on the SAME padded W.
    #[test]
    fn test_rok_bar_sum_m_3_pads_to_4() {
        let w = Mat::new(vec![
            vec![rq([1, 2, 0, 3])],
            vec![rq([0, 4, 1, 2])],
            vec![rq([5, 0, 1, 0])],
        ]);
        // t_vec must be computed from UNPADDED W (padded rows are 0 → no
        // contribution, so this is identical to computing from padded W).
        let t_vec = compute_t_vec(&w);
        let mut rng = rand::rng();
        let ((r_t, s_0), (_, s_1)) = rok_bar_sum::<Q, D>(1, 2, &t_vec, &w, &mut rng);
        assert_eq!(r_t.len(), 2, "l = ⌈log_2(3)⌉ = 2");
        assert_eq!(s_0.len(), 1);
        assert_eq!(s_1.len(), 1);
    }

    /// m=5 → pads to d_h^l = 8. Belt-and-braces companion to m=3.
    #[test]
    fn test_rok_bar_sum_m_5_pads_to_8() {
        let w = Mat::new(
            (0..5)
                .map(|i| vec![rq([i as u64, 0, 0, 0])])
                .collect::<Vec<_>>(),
        );
        let t_vec = compute_t_vec(&w);
        let mut rng = rand::rng();
        let ((r_t, _), _) = rok_bar_sum::<Q, D>(1, 2, &t_vec, &w, &mut rng);
        assert_eq!(r_t.len(), 3, "l = ⌈log_2(5)⌉ = 3");
    }

    /// m=1 → pads to 2. Edge case: single-row W.
    #[test]
    fn test_rok_bar_sum_m_1_pads_to_2() {
        let w = Mat::new(vec![vec![rq([1, 2, 0, 3])]]);
        let t_vec = compute_t_vec(&w);
        let mut rng = rand::rng();
        let ((r_t, _), _) = rok_bar_sum::<Q, D>(1, 2, &t_vec, &w, &mut rng);
        assert_eq!(r_t.len(), 1, "l = ⌈log_2(1)⌉ = 1 (min)");
    }

    /// t_vec.len() mismatch panics at the contract assert.
    #[test]
    #[should_panic]
    fn test_rok_bar_sum_wrong_t_len_panics() {
        let w = Mat::new(vec![
            vec![rq([1, 2, 0, 3]), rq([5, 0, 1, 0])],
            vec![rq([0, 4, 1, 2]), rq([3, 1, 0, 6])],
        ]);
        let bad_t = vec![R::zero(); 1]; // expected r = 2
        let mut rng = rand::rng();
        let _ = rok_bar_sum::<Q, D>(1, 2, &bad_t, &w, &mut rng);
    }

    // ─── pad_mat_to_d_exp ───

    /// Already-on-hypercube: target == m → no padding, returned matrix is
    /// equal (by ==) to input.
    #[test]
    fn test_pad_noop_when_already_power_of_d() {
        let w = mat(&[[1, 2], [3, 4]]); // m=2, d=2 → target=2
        let (padded, l) = pad_mat_to_d_exp(&w, 2);
        assert_eq!(l, 1);
        assert_eq!(padded.nrows(), 2);
        assert_eq!(padded, w);
    }

    /// m=3, d=2 → l=2, target=4. Pads 1 row of zeros at the bottom.
    /// Original rows preserved, padded row is all-zero polynomials.
    #[test]
    fn test_pad_m_3_d_2_appends_one_zero_row() {
        let w = mat(&[[1, 2], [3, 4], [5, 6]]);
        let (padded, l) = pad_mat_to_d_exp(&w, 2);
        assert_eq!(l, 2);
        assert_eq!(padded.nrows(), 4);
        assert_eq!(padded.ncols(), 2);
        // Original rows unchanged.
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(padded[(i, j)], w[(i, j)]);
            }
        }
        // Padded row is zero.
        for j in 0..2 {
            assert_eq!(padded[(3, j)], R::zero());
        }
    }

    /// m=1, d=2 → l=1, target=2. Single padded row.
    #[test]
    fn test_pad_m_1_pads_one_row() {
        let w = mat(&[[7, 8, 9]]);
        let (padded, l) = pad_mat_to_d_exp(&w, 2);
        assert_eq!(l, 1);
        assert_eq!(padded.nrows(), 2);
        for j in 0..3 {
            assert_eq!(padded[(1, j)], R::zero());
        }
    }

    /// d=3 (non-binary hypercube), m=5 → l=2, target=9. Pads 4 zero rows.
    #[test]
    fn test_pad_d_3_m_5() {
        let w = mat(&[[1], [2], [3], [4], [5]]);
        let (padded, l) = pad_mat_to_d_exp(&w, 3);
        assert_eq!(l, 2);
        assert_eq!(padded.nrows(), 9);
        for i in 5..9 {
            assert_eq!(padded[(i, 0)], R::zero());
        }
    }

    /// Padded W and unpadded W produce the same t_vec — required invariant for
    /// the comment in rok_bar_sum (t_vec computed pre-padding stays valid).
    #[test]
    fn test_pad_preserves_t_vec() {
        let w = mat(&[[1, 2], [3, 4], [5, 6]]); // m=3
        let (padded, _) = pad_mat_to_d_exp(&w, 2);
        assert_eq!(compute_t_vec(&w), compute_t_vec(&padded));
    }

    // NOTE on wrong-t soundness testing: perturbing t_vec by a constant
    // polynomial Δ contributes `Σ_{k=0..D/e} u^k · NTT(Δ)[k]` to `a_0`. This
    // is a degree-≤(D/e − 1) polynomial in u, so over Z_q it can have up to
    // D/e − 1 roots — and the test silently passes whenever `sample_u_vec`
    // picks one. For Q=17, D=4, e=1 the silent-pass set is the non-1 4-th
    // roots of unity {4, 13, 16} → ~19% flake rate. A robust soundness test
    // needs a fixed RNG seed (and a verified u) or a much larger Q; for now
    // `test_batched_slot_prover_wrong_claim_panics` covers the same soundness
    // path with a deterministic +1 perturbation on the claim itself.
}
