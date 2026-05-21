//! Multivariate sumcheck protocol over hypercube [D_h]^l.
//!
//! Reference: `06_salsaa/sumcheck.py`.
//!
//! NOTE: this implementation uses a **bookkeeping table** representation (one
//! evaluation per hypercube point), NOT the symbolic-polynomial substitution
//! used in the Python prototype. The bookkeeping version stores the function
//! as a flat Vec<Zq> of length D_h^num_vars and updates it in place each round,
//! avoiding the need for a symbolic multivariate polynomial type.
//!
//! Naming note: `D_h` (hypercube basis) is intentionally distinct from `D`
//! (cyclotomic dimension in `Rq<Q, D>`). They are unrelated constants.

use rand::Rng;

use crate::zq::Zq;

/// Sumcheck output: the verifier's running claim after the last round plus
/// the challenges chosen along the way. After receiving this, the verifier
/// still owes the final oracle check:
///     a_l ?= f(rands[0], ..., rands[l-1])
/// which lives OUTSIDE this routine (the caller must verify it via whatever
/// commitment / lookup is appropriate for f).
pub struct SumcheckOutput<const Q: u64> {
    pub a_l: Zq<Q>,
    pub rands: Vec<Zq<Q>>,
}

/// Sumcheck over hypercube [d_h]^num_vars for a function given by its
/// bookkeeping table.
///
/// Parameters:
/// - `book`: f evaluated at every point of [d_h]^num_vars. Length must equal
///           d_h^num_vars. Ordering: row-major with x_0 most significant
///           (i.e. index i has digit sequence (i / d_h^{l-1-j} mod d_h) for j-th variable).
/// - `claimed_sum`: a_0, prover's initial claim Σ_x f(x) = a_0.
/// - `num_vars`: l, number of variables.
/// - `d_h`: hypercube basis size (d_h = 2 for Boolean sumcheck).
/// - `rng`: verifier challenge source. (Replace with `&mut Transcript` once
///          Fiat–Shamir lands — see README Future.)
pub fn sumcheck<const Q: u64>(
    book: Vec<Zq<Q>>,
    claimed_sum: Zq<Q>,
    num_vars: usize,
    d_h: usize,
    rng: &mut impl Rng,
) -> SumcheckOutput<Q> {
    // Claim: Σ_{b_0} ... Σ_{b_{l-1}} f(b_0, ..., b_{l-1}) = a_0
    // a_j = the verifier's running claim before round j; a_0 = claimed_sum.

    // received_randoms = [r_0, r_1, ..., r_{j-1}] accumulated each round.

    // For each round j = 0..l:
    //
    //   Prover:
    //     Treat variable j as the symbolic X; sum the remaining variables over [d_h].
    //     g_j(X) = Σ_{b_{j+1}} ... Σ_{b_{l-1}} f(r_0, ..., r_{j-1}, X, b_{j+1}, ..., b_{l-1})
    //     Send g_j(0), g_j(1), ..., g_j(d_h - 1) to V.
    //     (d_h evals fully specify g_j since deg(g_j) ≤ d_h - 1.)
    //
    //   Verifier:
    //     Step 1: a_j ?= g_j(0) + g_j(1) + ... + g_j(d_h - 1).
    //     Step 2: sample r_j ← Z_q uniformly; set a_{j+1} = g_j(r_j); send r_j to P.
    //
    //   Prover:
    //     Update bookkeeping by collapsing variable j to r_j via lagrange interpolation
    //     over the d_h points {0, 1, ..., d_h - 1}. Resulting table has length / d_h.
    //
    // After l rounds, V holds a_l = g_{l-1}(r_{l-1}). V still owes:
    //   a_l ?= f(r_0, ..., r_{l-1})
    // — this final oracle check is the caller's responsibility (e.g. via an LDE
    //   evaluation or commitment opening).

    let _ = (book, claimed_sum, num_vars, d_h, rng);
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zq::Zq;

    const Q: u64 = 17;
    type F = Zq<Q>;

    /// Z_q element from a u64 (centered repr not needed here).
    fn zq(v: u64) -> F {
        F::new(v)
    }

    // ─── happy path: known small sum ───

    /// f(x_0, x_1) = x_0 + x_1 over [2]^2 hypercube.
    /// Hypercube points (x_0, x_1) in row-major order with x_0 MSB:
    ///   (0,0)→0  (0,1)→1  (1,0)→1  (1,1)→2
    /// Sum = 0 + 1 + 1 + 2 = 4.
    /// Sumcheck must (a) accept the correct claim, (b) return l=2 challenges.
    #[test]
    fn test_sumcheck_boolean_sum_of_two_vars() {
        let book = vec![zq(0), zq(1), zq(1), zq(2)];
        let claimed = zq(4);
        let mut rng = rand::rng();
        let out = sumcheck::<Q>(book, claimed, 2, 2, &mut rng);
        assert_eq!(out.rands.len(), 2, "one challenge per variable");
    }

    /// Wrong claim must blow up the protocol (sumcheck's Step 1 fails).
    #[test]
    #[should_panic]
    fn test_sumcheck_wrong_claim_panics() {
        let book = vec![zq(0), zq(1), zq(1), zq(2)]; // true sum is 4
        let bogus = zq(5);
        let mut rng = rand::rng();
        let _ = sumcheck::<Q>(book, bogus, 2, 2, &mut rng);
    }

    // ─── shape / API correctness ───

    /// Constant function f ≡ c over [d_h]^l: sum = d_h^l · c.
    #[test]
    fn test_sumcheck_constant_function() {
        // d_h = 3, l = 2  → hypercube has 9 points, all equal to c=2.  Sum = 9·2 = 18 = 1 mod 17.
        let book = vec![zq(2); 9];
        let claimed = zq(18 % 17);
        let mut rng = rand::rng();
        let out = sumcheck::<Q>(book, claimed, 2, 3, &mut rng);
        assert_eq!(out.rands.len(), 2);
    }
}
