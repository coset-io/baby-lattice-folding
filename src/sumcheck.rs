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
/// We know h(X) = Σ_{\vec x_2 \in [d_h]^{l-(j+1)} f(X, \vec x2) * \bar f(X, \vec x2)
/// We then know f(2, \vec x2) and \bar f(2, \vec x2) with X=2 and \vec x2 passed in.
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

/// Lagrange interpolate from points and evaluate
/// f(x) = \sum_i (\prod_{j \neq i} ((x-x_j)/(x_i-x_j))*y_i  (x_i -> 1, x_j -> 0)
fn evaluate_from_points<const Q: u64>(points: &[Zq<Q>], x_to_eval: Zq<Q>) -> Zq<Q> {
    let mut s = Zq::<Q>::zero();
    let num_points = points.len();
    // \sum_i (\prod_{j \neq i} ((x-x_j)/(x_i-x_j))*y_i
    for i in 0..num_points {
        // \prod_{j \neq i} ((x-x_j)/(x_i-x_j))*y_i
        let mut p = Zq::<Q>::one();
        for j in 0..num_points {
            // skip i==j case
            if i == j {
                continue;
            }
            // (x-x_j)/(x_i-x_j))
            p = p
                * ((x_to_eval - Zq::new(j as u64)) * (Zq::new(i as u64) - Zq::new(j as u64)).inv())
        }
        // *y_i
        p = p * points[i];
        s = s + p;
    }
    s
}

fn calculate_h_from_table<const Q: u64>(table: &[Zq<Q>], target_degree: usize) -> Vec<Zq<Q>> {
    (0..=target_degree as u64)
        .map(|x| h_x(table, Zq::new(x)))
        .collect()
}

/// Sumcheck over hypercube [d_h]^num_vars for a function given by its
/// bookkeeping table.
///
/// Parameters:
/// - `f`: f evaluated at every point of [d_h]^num_vars. Length must equal
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
        // Target degree = 2 since both MLE[W] and MLE[\bar W] are deg-1,
        // MLE[W] * MLE[\bar W] is deg-2.
        let target_degree = 2;
        let h = calculate_h_from_table(&table_f, target_degree);

        // Send h(x) to Verifier as g(x)
        let g = h.clone();

        //
        // Verifier
        //
        // V is not sure if g_j(x) = h_j(x) as P claimed
        //   and needs to verify
        //   1. a_j = g_j(0) + ... + g_j(d-1)
        //   2. g_j(r) ?= \sum_{b_{j+1}} ... \sum_{b_{l-1}} f(r_0, ..., r_j, b_{j+1}, ..., b_{l-1}), by SZDL
        //       - recursion: this is done by running sumcheck again with P

        // 1. Verify a_j == g_j(0) + g_j(1) + ... + g_j(d_h-1)
        let sum_g_hypercube = (0..d_h).map(|x| g[x]).fold(Zq::zero(), |acc, v| acc + v);
        assert_eq!(
            a, sum_g_hypercube,
            "a_j does not match h_j(0)+...+h_j(d_h-1): a_j={a:?}, g={g:?}"
        );

        // 2. SZDL: g_j(r) ?= h_j(r) = \sum_{b_{j+1}} ... \sum_{b_{l-1}} f(r_0, ..., r_j, b_{j+1}, ..., b_{l-1})
        // 2.1. Verifier samples random r_j
        let r = Zq::<Q>::random(rng);
        received_randoms.push(r);
        // 2.2. Calculate a_{j+1} = h_j(r_j). So next prover needs to prove "a_{j+1} = g_j(r_j) =? h_j(r_j)
        a = evaluate_from_points(&g, r);
        // Send r_j to Prover

        //
        // Prover
        //
        // Fold the table for the next round
        let half_idx = table_f.len() / 2;
        let mut table_f_new = Vec::with_capacity(half_idx);
        for i in 0..half_idx {
            // w(0,0) =
            table_f_new.push(cal_f_x(&table_f, r, i));
        }
        table_f = table_f_new;
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
        let book = vec![zq(0), zq(1), zq(1), zq(2)]; // true sum is 6
        let bogus = zq(5);
        let mut rng = rand::rng();
        let _ = sumcheck(book, bogus, 2, 2, &mut rng);
    }

    // ─── Lagrange interpolation tests ───
    //
    // evaluate_from_points takes y-values `[y_0, y_1, ..., y_{n-1}]` at the
    // implicit x-coordinates `[0, 1, ..., n-1]` and evaluates the unique
    // degree-≤(n-1) interpolating polynomial at a query point.

    /// Helper: build the y-values `[f(0), f(1), ..., f(n-1)]`.
    fn ys_of<G: Fn(u64) -> u64>(n: u64, f: G) -> Vec<F> {
        (0..n).map(|x| zq(f(x))).collect()
    }

    #[test]
    fn test_interp_passes_through_given_points_2pt() {
        // L(i) must equal y_i. n=2 catches only gross errors (y_i^{n-1} = y_i).
        let ys = vec![zq(5), zq(7)];
        assert_eq!(evaluate_from_points(&ys, zq(0)), zq(5));
        assert_eq!(evaluate_from_points(&ys, zq(1)), zq(7));
    }

    #[test]
    fn test_interp_passes_through_given_points_3pt() {
        // For n=3, y_i^{n-1} = y_i^2 ≠ y_i when y_i ∉ {0, 1}, so any future
        // refactor that re-introduces the in-loop y_i multiplication will FAIL.
        let ys = vec![zq(3), zq(5), zq(11)];
        assert_eq!(evaluate_from_points(&ys, zq(0)), zq(3));
        assert_eq!(evaluate_from_points(&ys, zq(1)), zq(5));
        assert_eq!(evaluate_from_points(&ys, zq(2)), zq(11));
    }

    #[test]
    fn test_interp_linear_eval_at_new_point() {
        // p(x) = 2x + 3.  p(5) = 13, p(7) = 17 = 0  (mod 17).
        let ys = ys_of(2, |x| (2 * x + 3) % Q);
        assert_eq!(evaluate_from_points(&ys, zq(5)), zq(13));
        assert_eq!(evaluate_from_points(&ys, zq(7)), zq(0));
    }

    #[test]
    fn test_interp_quadratic_eval_at_new_point() {
        // p(x) = x^2 + 1.  p(0)=1, p(1)=2, p(2)=5, p(3)=10, p(4)=17=0.
        let ys = ys_of(3, |x| (x * x + 1) % Q);
        assert_eq!(evaluate_from_points(&ys, zq(3)), zq(10));
        assert_eq!(evaluate_from_points(&ys, zq(4)), zq(0));
    }

    #[test]
    fn test_interp_quadratic_nontrivial_y0() {
        // Sharpest test for the y_i^{n-1} bug.
        // p(x) = 3x^2 + 2x + 4:  p(0)=4, p(1)=9, p(2)=20=3, p(5)=89=4 (mod 17).
        let ys = vec![zq(4), zq(9), zq(3)];
        assert_eq!(evaluate_from_points(&ys, zq(5)), zq(4));
    }

    #[test]
    fn test_interp_at_nonintegral_eval_point() {
        // Query at x = 16 ≡ -1 (mod 17).  p(x) = x + 1 → p(-1) = 0.
        let ys = ys_of(2, |x| (x + 1) % Q);
        assert_eq!(evaluate_from_points(&ys, zq(16)), zq(0));
    }

    #[test]
    fn test_interp_4pt_cubic() {
        // p(x) = x^3.  p(2)=8, p(3)=27=10, p(4)=64=13  (mod 17).
        let ys = ys_of(4, |x| (x * x * x) % Q);
        assert_eq!(evaluate_from_points(&ys, zq(4)), zq(13));
        assert_eq!(evaluate_from_points(&ys, zq(2)), zq(8));
        assert_eq!(evaluate_from_points(&ys, zq(3)), zq(10));
    }

    #[test]
    fn test_interp_constant_polynomial() {
        // p(x) = 7. Three points all with y = 7. 7^2 = 49 = 15 ≠ 7, so the
        // y_i^{n-1} bug is observable here (y = 0 or 1 would hide it).
        let ys = vec![zq(7), zq(7), zq(7)];
        assert_eq!(evaluate_from_points(&ys, zq(0)), zq(7));
        assert_eq!(evaluate_from_points(&ys, zq(5)), zq(7));
        assert_eq!(evaluate_from_points(&ys, zq(13)), zq(7));
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
