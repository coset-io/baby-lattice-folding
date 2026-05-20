//! SALSAA relation types.
//!
//! Σ^lin: ((H, F, Y, β), W) ∈ Σ^lin
//!   iff  H · F · W = Y  mod q
//!   and  max_i ‖w_i‖₂ ≤ β, i \in [r]
//!
//! where F = F_com stacked over F_eval, and ‖·‖₂ is the column l_2 norm
//! over the concatenated coefficient vector of each column of W.
//!
//! Reference: SALSAA paper §3–4. Python: `06_salsaa/relations.py`.

use crate::mat::Mat;
use crate::ring::Rq;

/// Public statement of a linear relation. Owns no witness data.
///
/// Dimensions (rows × cols):
///   H      : n̂ × n
///   F_com  : n̄ × m   (commitment block)
///   F_eval : ñ × m   (evaluation block; may be empty initially)
///   Y      : n̂ × r
///   β      : column l_2 norm bound
#[derive(Debug, Clone)]
pub struct LinInstance<const Q: u64, const D: usize> {
    pub h: Mat<Rq<Q, D>>,
    pub f_com: Mat<Rq<Q, D>>,
    pub f_eval: Mat<Rq<Q, D>>,
    pub y: Mat<Rq<Q, D>>,
    pub beta: u64,
}

impl<const Q: u64, const D: usize> LinInstance<Q, D> {
    pub fn new(
        h: Mat<Rq<Q, D>>,
        f_com: Mat<Rq<Q, D>>,
        f_eval: Mat<Rq<Q, D>>,
        y: Mat<Rq<Q, D>>,
        beta: u64,
    ) -> Self {
        // n_hat
        assert_eq!(h.nrows(), y.nrows());
        // n
        assert_eq!(h.ncols(), f_com.nrows() + f_eval.nrows());
        // m
        assert_eq!(f_com.ncols(), f_eval.ncols());
        Self {
            h,
            f_com,
            f_eval,
            y,
            beta,
        }
    }

    /// F = F_com stacked on top of F_eval.
    pub fn f(&self) -> Mat<Rq<Q, D>> {
        self.f_com.stack(&self.f_eval)
    }

    /// n̂ — rows of H (output dimension after H).
    pub fn n_hat(&self) -> usize {
        self.h.nrows()
    }

    /// n — cols of H = rows of F.
    pub fn n(&self) -> usize {
        self.h.ncols()
    }

    /// n_top — cols of H = rows of F.
    pub fn n_top(&self) -> usize {
        self.f_com.nrows()
    }

    /// m — cols of F = rows of W.
    pub fn m(&self) -> usize {
        self.f_com.ncols()
    }

    /// r — cols of Y = cols of W (number of stacked witnesses).
    pub fn r(&self) -> usize {
        self.y.ncols()
    }

    /// Append rows to F_eval and Y, extending H with an identity block.
    ///
    /// new H = [[H,  0],
    ///          [0,  I]]
    pub fn with_extra_eval(&self, _new_f_rows: Mat<Rq<Q, D>>, _new_y_rows: Mat<Rq<Q, D>>) -> Self {
        todo!()
    }
}

/// Private witness W of shape m × r over R_q.
#[derive(Debug, Clone)]
pub struct LinWitness<const Q: u64, const D: usize> {
    pub w: Mat<Rq<Q, D>>,
}

impl<const Q: u64, const D: usize> LinWitness<Q, D> {
    pub fn new(w: Mat<Rq<Q, D>>) -> Self {
        Self { w }
    }

    /// m — rows of W.
    pub fn m(&self) -> usize {
        self.w.nrows()
    }

    /// r — cols of W (number of witnesses).
    pub fn r(&self) -> usize {
        self.w.ncols()
    }
}

/// A LinInstance plus a verified LinWitness. Construction must enforce:
///   1. Dimension consistency (H, F, W, Y all line up)
///   2. Algebraic relation:   H · (F · W) = Y
///   3. l_2 norm bound:        max_i ‖w_i‖₂ ≤ β
///
/// `new` returns `Self` here for symmetry with Python. You can refactor
/// to `Result<Self, RelError>` (Ch 9 practice) if you want explicit
/// error variants instead of panic.
#[derive(Debug, Clone)]
pub struct LinRelation<const Q: u64, const D: usize> {
    pub instance: LinInstance<Q, D>,
    pub witness: LinWitness<Q, D>,
}

fn col_l2_norm_squared<const Q: u64, const D: usize>(col: &[Rq<Q, D>]) -> u64 {
    col.iter().map(|r| r.l2_norm_squared()).sum()
}

fn max_col_l2_norm_squared<const Q: u64, const D: usize>(w: &Mat<Rq<Q, D>>) -> u64 {
    (0..w.ncols())
        .map(|j| col_l2_norm_squared(&w.col(j)))
        .max()
        .unwrap_or(0)
}

impl<const Q: u64, const D: usize> LinRelation<Q, D> {
    pub fn new(instance: LinInstance<Q, D>, witness: LinWitness<Q, D>) -> Self {
        // 1. l_2 norm of W must be <= \beta
        let l2_norm_squared_w = max_col_l2_norm_squared(&witness.w);
        let l2_norm_bound_squared = instance.beta * instance.beta;
        assert!(
            l2_norm_squared_w < l2_norm_bound_squared,
            "exceeded norm bound: actual norm squared={}, norm bound={}",
            l2_norm_squared_w,
            l2_norm_bound_squared,
        );

        // 2. Verify HFW = Y, where H, F, Y from both sides and W only from Prover
        let lhs = instance.h.clone() * instance.f() * witness.w.clone();
        assert_eq!(
            lhs,
            instance.y,
            "relation doesn't hold:\n  H={:?}\n  F={:?}\n  W={:?}\n  Y={:?}\n  H·F·W={:?}",
            instance.h,
            instance.f(),
            witness.w,
            instance.y,
            lhs
        );
        Self { instance, witness }
    }

    pub fn n_hat(&self) -> usize {
        self.instance.n_hat()
    }

    pub fn n(&self) -> usize {
        self.instance.n()
    }

    /// n_top = F_com.nrows(). Useful later for the join / batch RoKs.
    pub fn n_top(&self) -> usize {
        self.instance.n_top()
    }

    pub fn m(&self) -> usize {
        self.instance.m()
    }

    pub fn r(&self) -> usize {
        self.instance.r()
    }

    pub fn beta(&self) -> u64 {
        self.instance.beta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    // NOTE: Initial Σ^lin has F_eval = 0 × m. Current `Mat::new` requires
    // at least one row, so 0-row matrices can't be constructed yet. All
    // tests below use F_eval with ≥ 1 row to dodge that limitation. When
    // you decide how to handle the empty case (Option, Mat::empty, etc.),
    // add a test for the initial state.

    // ─── LinWitness ───

    #[test]
    fn test_witness_new_dims() {
        let w = mat(&[[1], [2], [3]]); // 3 × 1
        let lw = LinWitness::new(w);
        assert_eq!(lw.m(), 3);
        assert_eq!(lw.r(), 1);
    }

    // ─── LinInstance ───

    #[test]
    fn test_instance_dim_accessors() {
        // H: 2×2, F_com: 1×3, F_eval: 1×3 → F: 2×3, Y: 2×4
        let h = Mat::<R>::identity(2);
        let f_com = mat(&[[1, 2, 3]]);
        let f_eval = mat(&[[4, 5, 6]]);
        let y = mat(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
        let inst = LinInstance::new(h, f_com, f_eval, y, 10_000);
        assert_eq!(inst.n_hat(), 2);
        assert_eq!(inst.n(), 2);
        assert_eq!(inst.m(), 3);
        assert_eq!(inst.r(), 4);
    }

    #[test]
    fn test_instance_f_is_f_com_stacked_on_f_eval() {
        let h = Mat::<R>::identity(2);
        let f_com = mat(&[[1, 2]]); // 1 × 2 → top
        let f_eval = mat(&[[3, 4]]); // 1 × 2 → bottom
        let y = Mat::<R>::zero(2, 1);
        let inst = LinInstance::new(h, f_com, f_eval, y, 10_000);
        let f = inst.f();
        assert_eq!(f.dimensions(), (2, 2));
        assert_eq!(f[(0, 0)], c(1));
        assert_eq!(f[(0, 1)], c(2));
        assert_eq!(f[(1, 0)], c(3));
        assert_eq!(f[(1, 1)], c(4));
    }

    #[test]
    #[should_panic]
    fn test_instance_f_com_f_eval_width_mismatch_panics() {
        // F_com.ncols=3 vs F_eval.ncols=2 → invalid
        let h = Mat::<R>::identity(2);
        let f_com = mat(&[[1, 2, 3]]);
        let f_eval = mat(&[[4, 5]]);
        let y = Mat::<R>::zero(2, 1);
        let _ = LinInstance::new(h, f_com, f_eval, y, 10_000);
    }

    // ─── LinRelation ───

    /// Tiny satisfied relation:
    ///   H = I_3
    ///   F_com = [[1, 2], [3, 4]]    (2 × 2)
    ///   F_eval = [[5, 6]]            (1 × 2)
    ///   F = stack → 3 × 2
    ///   W = [[1], [1]]               (2 × 1)
    ///   F · W = [3, 7, 11]^T (constant-polynomial entries)
    ///   Y = H · (F · W) = F · W  since H = I
    fn small_valid_rel_components() -> (Mat<R>, Mat<R>, Mat<R>, Mat<R>, Mat<R>) {
        let h = Mat::<R>::identity(3);
        let f_com = mat(&[[1, 2], [3, 4]]);
        let f_eval = mat(&[[5, 6]]);
        let w = mat(&[[1], [1]]);
        let y = mat(&[[3], [7], [11]]);
        (h, f_com, f_eval, w, y)
    }

    #[test]
    fn test_relation_valid_constructs() {
        let (h, f_com, f_eval, w, y) = small_valid_rel_components();
        let inst = LinInstance::new(h, f_com, f_eval, y, 10_000);
        let wit = LinWitness::new(w);
        let _rel = LinRelation::new(inst, wit);
    }

    #[test]
    fn test_relation_accessors() {
        let (h, f_com, f_eval, w, y) = small_valid_rel_components();
        let inst = LinInstance::new(h, f_com, f_eval, y, 42);
        let wit = LinWitness::new(w);
        let rel = LinRelation::new(inst, wit);
        assert_eq!(rel.n_hat(), 3);
        assert_eq!(rel.n(), 3);
        assert_eq!(rel.n_top(), 2); // F_com has 2 rows
        assert_eq!(rel.m(), 2);
        assert_eq!(rel.r(), 1);
        assert_eq!(rel.beta(), 42);
    }

    #[test]
    #[should_panic]
    fn test_relation_invariant_violation_panics() {
        // Same setup but Y is wrong (all zeros instead of [3, 7, 11]).
        let h = Mat::<R>::identity(3);
        let f_com = mat(&[[1, 2], [3, 4]]);
        let f_eval = mat(&[[5, 6]]);
        let w = mat(&[[1], [1]]);
        let wrong_y = Mat::<R>::zero(3, 1);

        let inst = LinInstance::new(h, f_com, f_eval, wrong_y, 10_000);
        let wit = LinWitness::new(w);
        let _ = LinRelation::new(inst, wit);
    }

    #[test]
    #[should_panic]
    fn test_relation_dim_mismatch_panics() {
        // W has wrong nrows: F.ncols = 2 but W.nrows = 3.
        let h = Mat::<R>::identity(3);
        let f_com = mat(&[[1, 2], [3, 4]]);
        let f_eval = mat(&[[5, 6]]);
        let bad_w = mat(&[[1], [1], [1]]); // 3 × 1 instead of 2 × 1
        let y = mat(&[[3], [7], [11]]);

        let inst = LinInstance::new(h, f_com, f_eval, y, 10_000);
        let wit = LinWitness::new(bad_w);
        let _ = LinRelation::new(inst, wit);
    }

    // NOTE: norm-bound violation test (||w_i||_2 > β) is deferred — it
    // needs `Zq::centered()` (mapping v ∈ [0, q) → signed [-q/2, q/2])
    // which isn't on Zq yet. Once that exists, add a `should_panic` test
    // putting a column of large-coefficient entries into W with β = 1.
}
