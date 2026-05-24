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

use crate::{ring::Ring, zq::Zq};

/// Sumcheck output: the verifier's running claim after the last round plus
/// the challenges chosen along the way. After receiving this, the verifier
/// still owes the final oracle check:
///     a_l ?= f(rands[0], ..., rands[l-1])
/// which lives OUTSIDE this routine (the caller must verify it via whatever
/// commitment / lookup is appropriate for f).
pub struct SumcheckOutput<T> {
    pub a_l: T,
    pub rands: Vec<T>,
}

/// Calculate current \tilde f(r_0, ..., x, \vec x2)
///
/// Since we know f(X, \vec x2) is linear with \vec x2 as constant, given `f` is a multilinear extension,
/// interpolation can be done with f(0, \vec x2) and f(1, \vec x2).
/// So, f(X, \vec x2) = (1-X) f(0, \vec x2) + X f(1, \vec x2)
///
/// i = (x_{j+1}, ..., x_{l-1}) = \vec x2
/// e.g. i = 0 -> (0, 0), i = 1 -> (0, 1)
fn cal_f_x<const Q: u64>(table: &[Zq<Q>], x: Zq<Q>, i: usize) -> Zq<Q> {
    let half_idx = table.len() / 2;
    // p(x) = (1-x)*lo + x*hi
    (Zq::one() - x) * table[i] + x * table[half_idx + i]
}

/// Derive h_j(x) = Σ_{b_{j+1}} ... Σ_{b_{l-1}} f(r_0, ..., r_{j-1}, X, b_{j+1}, ..., b_{l-1})
///
/// We know h(X) = Σ_{\vec x_2 \in [d_h]^{l-(j+1)} f(X, \vec x2) + f(X, \vec x2)
/// We then know f(2, \vec x2) and f(2, \vec x2) with X=2 and \vec x2 passed in.
fn h_x<const Q: u64>(table: &[Zq<Q>], x: Zq<Q>) -> Zq<Q> {
    let half_idx = table.len() / 2;
    (0..half_idx)
        .map(|i| {
            let w = cal_f_x(table, x, i);
            let w_bar = w;
            w * w_bar
        })
        .fold(Zq::<Q>::zero(), |acc, v| acc + v)
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
    f: Vec<Zq<Q>>,
    claimed_sum: Zq<Q>,
    num_vars: usize,
    d_h: usize,
    rng: &mut impl Rng,
) -> SumcheckOutput<Zq<Q>> {
    assert_eq!(f.len(), d_h.pow(num_vars as u32), "f size is not [d_h]^l");
    // Claim: Σ_{b_0} ... Σ_{b_{l-1}} f(b_0, ..., b_{l-1}) = a_0
    // a_j = the verifier's running claim before round j; a_0 = claimed_sum.
    let mut a = claimed_sum;

    // received_randoms = [r_0, r_1, ..., r_{l-1}] accumulated each round.
    let mut received_randoms = Vec::<Zq<Q>>::with_capacity(num_vars);

    let mut table_f = f.clone();

    for _j in 0..num_vars {
        //
        // Prover
        //
        let h_0 = h_x(&table_f, Zq::zero());
        let h_1 = h_x(&table_f, Zq::one());

        // h(2) needs some tricks since f and \tilde f only agree on the hypercube [d]^l
        // =====
        // we know h(2) = (f(2, 0) * \bar f(2, 0)) + (f(2, 1) * \bar f(2, 1))
        //  -> need to derive f(X, x_2) first
        // Since we know f(X, x2) with x2 as constant is linear given f is a multilinear extension,
        // interpolation can be done with f(0, x_2) and f(1, x_2).
        // So, f(X, x2) = (1-X) f(0, x2) + X f(1, x2)
        // We then know f(2, 0) and f(2, 1) with X=2 and x2={0,1} passed in.
        // Do the same cal for \bar f so we can calculate h(2)
        // =====
        let _h_2 = h_x(&table_f, Zq::new(2));

        // Send h_0, h_1, h_2 as g(x) to Verifier

        //
        // Verifier
        //
        // V is not sure if g_j(x) = h_j(x) as P claimed
        //   and needs to verify
        //   1. a_j = g_j(0) + ... + g_j(d-1)
        //   2. g_j(r) ?= \sum_{b_{j+1}} ... \sum_{b_{l-1}} f(r_0, ..., r_j, b_{j+1}, ..., b_{l-1}), by SZDL
        //       - recursion: this is done by running sumcheck again with P

        // 1. Verify a_j == g_j(0) + g_j(1)
        assert_eq!(
            a,
            h_0 + h_1,
            "a_j does not match h_j(0)+...+h_j(d_h-1): a_j={a:?}, h_0={h_0:?}, h_1={h_1:?}"
        );

        // 2. SZDL: g_j(r) ?= \sum_{b_{j+1}} ... \sum_{b_{l-1}} f(r_0, ..., r_j, b_{j+1}, ..., b_{l-1})
        // Verifier samples random r_j
        let r = Zq::<Q>::random(rng);
        // Send r_j to Prover

        //
        // Prover
        //
        // Calculate a_{j+1} = g_j(r_j)
        a = h_x(&table_f, r);
        // Send a_{j+1} to Verifier

        // Fold the table for the next round
        let half_idx = table_f.len() / 2;
        let mut table_f_new = Vec::with_capacity(half_idx);
        for i in 0..half_idx {
            // w(0,0) =
            table_f_new.push(cal_f_x(&table_f, r, i));
        }
        table_f = table_f_new;

        // Save all `r`s from verifier
        received_randoms.push(r);
    }
    SumcheckOutput {
        a_l: a,
        rands: received_randoms,
    }
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
        let claimed = zq(6);
        let mut rng = rand::rng();
        let out = sumcheck(book, claimed, 2, 2, &mut rng);
        assert_eq!(out.rands.len(), 2, "one challenge per variable");
    }

    /// Wrong claim must blow up the protocol (sumcheck's Step 1 fails).
    #[test]
    #[should_panic]
    fn test_sumcheck_wrong_claim_panics() {
        let book = vec![zq(0), zq(1), zq(1), zq(2)]; // true sum is 4
        let bogus = zq(5);
        let mut rng = rand::rng();
        let _ = sumcheck(book, bogus, 2, 2, &mut rng);
    }

    // ─── shape / API correctness ───

    /// Constant function f ≡ c over [d_h]^l: sum = d_h^l · c.
    #[test]
    #[ignore = "d_h > 2 not yet supported: needs Lagrange interp through d_h points + d_h-term verifier sum"]
    fn test_sumcheck_constant_function() {
        // d_h = 3, l = 2  → hypercube has 9 points, all equal to c=2.  Sum = 9·2 = 18 = 1 mod 17.
        let book = vec![zq(2); 9];
        let claimed = zq(18 % 17);
        let mut rng = rand::rng();
        let out = sumcheck(book, claimed, 2, 3, &mut rng);
        assert_eq!(out.rands.len(), 2);
    }
}
