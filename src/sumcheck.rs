use rand::Rng;

use crate::zq::Zq;

/// Sumcheck output: the verifier's running claim after the last round plus
/// the challenges chosen along the way. After receiving this, the verifier
/// still owes the final oracle check:
///     a_l ?= f(rands[0], ..., rands[l-1])
/// which lives OUTSIDE this routine (the caller can call
/// `prover.final_value()` and compare, or — in a real protocol — open a PCS
/// commitment at `rands`).
pub struct SumcheckOutput<T> {
    pub a_l: T,
    pub rands: Vec<T>,
}

/// A prover that can drive the sumcheck protocol over `Z_q`.
///
/// The sumcheck driver is generic over this trait. Different protocols
/// (single-product, batched-slot RLC, R1CS-style, ...) each provide their own
/// impl, while the driver's structure stays the same.
pub trait SumcheckProver<const Q: u64> {
    /// Round message for the current variable being processed.
    ///
    /// Returns `[g(0), g(1), ..., g(target_degree)]` — `target_degree + 1`
    /// evaluations of the round polynomial g. The driver doesn't need to know
    /// `target_degree`; it reads `g.len()` and feeds it to Lagrange interp.
    fn round_message(&self) -> Vec<Zq<Q>>;

    /// Collapse internal bookkeeping state using verifier's challenge `r`.
    /// After this call, the prover represents the function with one fewer
    /// variable (the leading variable is bound to `r`).
    fn fold(&mut self, r: Zq<Q>);

    /// After all variables have been folded, return `\tilde f(rands)` — the
    /// final value the verifier owes a separate oracle check for. For
    /// [`SingleProductProver`] this is `MLE[A](rands) · MLE[B](rands)`;
    /// future batched impls will RLC their per-slot final values here.
    fn final_value(&self) -> Zq<Q>;
}

/// Prover for `Σ_x A(x) · B(x)` over `[d_h]^l`, where A and B are multilinear
/// extensions given by their bookkeeping tables on the hypercube.
///
/// This is the educational baseline — the `r · d/e = 1` special case of the
/// SALSAA norm-check sumcheck. Each round's round-poly has degree 2 (product
/// of two degree-1 MLEs), so 3 evaluations are sent per round.
pub struct SingleProductProver<const Q: u64> {
    table_a: Vec<Zq<Q>>,
    table_b: Vec<Zq<Q>>,
}

impl<const Q: u64> SingleProductProver<Q> {
    pub fn new(table_a: Vec<Zq<Q>>, table_b: Vec<Zq<Q>>) -> Self {
        assert_eq!(
            table_a.len(),
            table_b.len(),
            "table_a and table_b must have the same length"
        );
        Self { table_a, table_b }
    }
}

impl<const Q: u64> SumcheckProver<Q> for SingleProductProver<Q> {
    fn round_message(&self) -> Vec<Zq<Q>> {
        // MLE[A] · MLE[B] is degree-2 per variable → 3 evaluations per round.
        const TARGET_DEGREE: usize = 2;
        calculate_h_from_table(&self.table_a, &self.table_b, TARGET_DEGREE)
    }

    fn fold(&mut self, r: Zq<Q>) {
        self.table_a = fold_table(&self.table_a, r);
        self.table_b = fold_table(&self.table_b, r);
    }

    fn final_value(&self) -> Zq<Q> {
        // After all rounds, each table is length 1 = MLE evaluated at rands.
        assert_eq!(
            self.table_a.len(),
            1,
            "final_value() called before sumcheck fully folded (table len = {})",
            self.table_a.len(),
        );
        self.table_a[0] * self.table_b[0]
    }
}

/// Calculate current \tilde f(r_0, ..., x, \vec x2)
///
/// Since we know f(X, \vec x2) is linear with \vec x2 as constant, given `f` is a multilinear extension,
/// interpolation can be done with f(0, \vec x2) and f(1, \vec x2).
/// So, f(X, \vec x2) = (1-X) f(0, \vec x2) + X f(1, \vec x2)
///
/// i = (x_{j+1}, ..., x_{l-1}) = \vec x2
/// e.g. i = 0 -> (0, 0), i = 1 -> (0, 1)
pub fn cal_f_x<const Q: u64>(table: &[Zq<Q>], x: Zq<Q>, i: usize) -> Zq<Q> {
    let half_idx = table.len() / 2;
    // p(x) = (1-x)*lo + x*hi
    (Zq::one() - x) * table[i] + x * table[half_idx + i]
}

/// Derive h_j(x) = Σ_{b_{j+1}} ... Σ_{b_{l-1}} f(r_0, ..., r_{j-1}, X, b_{j+1}, ..., b_{l-1}) · \bar f(...)
///
/// h(X) = Σ_{\vec x_2 \in [d_h]^{l-(j+1)}} f(X, \vec x2) * \bar f(X, \vec x2)
/// We extrapolate f and \bar f to X separately via the multilinear formula,
/// then multiply and sum over remaining hypercube variables.
pub fn h_x<const Q: u64>(table_f: &[Zq<Q>], table_f_bar: &[Zq<Q>], x: Zq<Q>) -> Zq<Q> {
    let half_idx = table_f.len() / 2;
    (0..half_idx)
        .map(|i| {
            let w = cal_f_x(table_f, x, i);
            let w_bar = cal_f_x(table_f_bar, x, i);
            w * w_bar
        })
        .fold(Zq::<Q>::zero(), |acc, v| acc + v)
}

/// Lagrange interpolate from points and evaluate.
///
/// `points = [y_0, y_1, ..., y_{n-1}]` are the values at x-coordinates
/// `0, 1, ..., n-1` (implicit). Returns the unique degree-≤(n-1) interpolant
/// evaluated at `x_to_eval`.
fn evaluate_from_points<const Q: u64>(points: &[Zq<Q>], x_to_eval: Zq<Q>) -> Zq<Q> {
    let mut s = Zq::<Q>::zero();
    let num_points = points.len();
    // L(x) = Σ_i y_i · Π_{j ≠ i} (x - j) / (i - j)
    for (i, &y_i) in points.iter().enumerate() {
        // p = Π_{j ≠ i} (x - j) / (i - j)  -- pure barycentric basis
        let mut p = Zq::<Q>::one();
        for j in 0..num_points {
            if i == j {
                continue;
            }
            p = p
                * ((x_to_eval - Zq::new(j as u64)) * (Zq::new(i as u64) - Zq::new(j as u64)).inv())
        }
        // multiply y_i exactly once, OUTSIDE the j-loop
        p = p * y_i;
        s = s + p;
    }
    s
}

fn calculate_h_from_table<const Q: u64>(
    table_f: &[Zq<Q>],
    table_f_bar: &[Zq<Q>],
    target_degree: usize,
) -> Vec<Zq<Q>> {
    (0..=target_degree as u64)
        .map(|x| h_x(table_f, table_f_bar, Zq::new(x)))
        .collect()
}

pub fn fold_table<const Q: u64>(table: &[Zq<Q>], r: Zq<Q>) -> Vec<Zq<Q>> {
    let half_idx = table.len() / 2;
    (0..half_idx).map(|i| cal_f_x(table, r, i)).collect()
}

/// Sumcheck driver. Generic over the prover — all protocol-level work
/// (round-message verification, Lagrange interpolation, challenge sampling)
/// lives here; per-protocol specialisation lives in the `SumcheckProver` impl.
///
/// Parameters:
/// - `prover`: prover state implementing [`SumcheckProver`].
/// - `claimed_sum`: a_0, prover's initial claim Σ_x \tilde f(x) = a_0.
/// - `num_vars`: l, number of variables in the hypercube.
/// - `d_h`: hypercube basis size (d_h = 2 for Boolean sumcheck).
/// - `rng`: verifier challenge source. (Replace with `&mut Transcript` once
///   Fiat–Shamir lands — see README Future.)
pub fn sumcheck<const Q: u64, P: SumcheckProver<Q>>(
    prover: &mut P,
    claimed_sum: Zq<Q>,
    num_vars: usize,
    d_h: usize,
    rng: &mut impl Rng,
) -> SumcheckOutput<Zq<Q>> {
    let mut a = claimed_sum;
    let mut received_randoms = Vec::<Zq<Q>>::with_capacity(num_vars);

    for _j in 0..num_vars {
        // Prover sends g(X) = h_j(X) to Verifier
        let g = prover.round_message();

        // Verifier check 1: a == g(0) + g(1) + ... + g(d_h - 1)
        let sum_g_hypercube = (0..d_h).map(|x| g[x]).fold(Zq::zero(), |acc, v| acc + v);
        assert_eq!(
            a, sum_g_hypercube,
            "round-message claim mismatch: a={a:?}, g={g:?}"
        );

        // Verifier samples r_j and updates the running claim via SZDL.
        let r = Zq::<Q>::random(rng);
        received_randoms.push(r);
        a = evaluate_from_points(&g, r);

        // Prover folds internal state for the next round.
        prover.fold(r);
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

    // ─── End-to-end sumcheck: SingleProductProver ───

    /// f(x_0, x_1) = x_0 + x_1 over [2]^2 hypercube.
    /// Hypercube points (x_0, x_1) in row-major order with x_0 MSB:
    ///   (0,0)→0  (0,1)→1  (1,0)→1  (1,1)→2
    /// With A = B = f, the claim is Σ f(i)² = 0 + 1 + 1 + 4 = 6.
    #[test]
    fn test_sumcheck_boolean_sum_of_squares() {
        let table = vec![zq(0), zq(1), zq(1), zq(2)];
        let claimed = zq(6);
        let mut rng = rand::rng();
        let mut prover = SingleProductProver::new(table.clone(), table);
        let out = sumcheck(&mut prover, claimed, 2, 2, &mut rng);
        assert_eq!(out.rands.len(), 2, "one challenge per variable");
    }

    /// Wrong claim must trigger the round-0 check.
    #[test]
    #[should_panic(expected = "round-message claim mismatch")]
    fn test_sumcheck_wrong_claim_panics() {
        let table = vec![zq(0), zq(1), zq(1), zq(2)]; // true Σ f² = 6
        let bogus = zq(5);
        let mut rng = rand::rng();
        let mut prover = SingleProductProver::new(table.clone(), table);
        let _ = sumcheck(&mut prover, bogus, 2, 2, &mut rng);
    }

    /// **End-to-end correctness**: after sumcheck completes honestly,
    /// `out.a_l` must equal `MLE[A](rands) · MLE[B](rands)`.
    ///
    /// This is the test that catches fold-correctness bugs: if either table
    /// isn't being folded properly each round, the final `a_l` won't agree
    /// with the manually-computed MLE evaluation.
    #[test]
    fn test_sumcheck_a_l_matches_mle_product() {
        // Pick A and B independently — important so the test detects bugs
        // where the two tables get muddled (e.g. only one is folded).
        let a_table = vec![zq(3), zq(7), zq(5), zq(11)];
        let b_table = vec![zq(2), zq(4), zq(8), zq(13)];

        // True claim: Σ_i a[i]·b[i]
        // = 3·2 + 7·4 + 5·8 + 11·13
        // = 6 + 28 + 40 + 143 = 217 ≡ 217 - 12·17 = 217 - 204 = 13 (mod 17)
        let claimed: F = a_table
            .iter()
            .zip(&b_table)
            .map(|(&a, &b)| a * b)
            .fold(zq(0), |acc, v| acc + v);
        assert_eq!(claimed, zq(13), "sanity: inner product should be 13");

        let mut rng = rand::rng();
        let mut prover = SingleProductProver::new(a_table.clone(), b_table.clone());
        let out = sumcheck(&mut prover, claimed, 2, 2, &mut rng);

        // Independently fold the original tables with the same challenges to
        // get MLE[A](rands) and MLE[B](rands).
        let mut a_mle = a_table;
        let mut b_mle = b_table;
        for &r in &out.rands {
            a_mle = fold_table(&a_mle, r);
            b_mle = fold_table(&b_mle, r);
        }
        assert_eq!(a_mle.len(), 1);
        assert_eq!(b_mle.len(), 1);

        let expected = a_mle[0] * b_mle[0];
        assert_eq!(
            out.a_l, expected,
            "a_l should equal MLE[A](rands) · MLE[B](rands)"
        );

        // Cross-check: prover's own final_value() should agree.
        assert_eq!(
            prover.final_value(),
            expected,
            "prover.final_value() should match independent MLE evaluation"
        );
    }

    /// Same as above but l=3 (8-point hypercube) — catches off-by-one in
    /// fold loops more aggressively.
    #[test]
    fn test_sumcheck_a_l_matches_mle_product_l3() {
        let a_table: Vec<F> = (0..8u64).map(|i| zq(i + 1)).collect();
        let b_table: Vec<F> = (0..8u64).map(|i| zq(i * 3 + 2)).collect();
        let claimed: F = a_table
            .iter()
            .zip(&b_table)
            .map(|(&a, &b)| a * b)
            .fold(zq(0), |acc, v| acc + v);

        let mut rng = rand::rng();
        let mut prover = SingleProductProver::new(a_table.clone(), b_table.clone());
        let out = sumcheck(&mut prover, claimed, 3, 2, &mut rng);

        let mut a_mle = a_table;
        let mut b_mle = b_table;
        for &r in &out.rands {
            a_mle = fold_table(&a_mle, r);
            b_mle = fold_table(&b_mle, r);
        }
        assert_eq!(out.a_l, a_mle[0] * b_mle[0]);
        assert_eq!(prover.final_value(), a_mle[0] * b_mle[0]);
    }

    /// d_h > 2 not supported by SingleProductProver's hardcoded
    /// `TARGET_DEGREE = 2`. Kept ignored as a future-work marker.
    #[test]
    #[ignore = "d_h > 2 needs target_degree = 2(d_h - 1); not yet wired"]
    fn test_sumcheck_constant_function() {
        let table = vec![zq(2); 9];
        let claimed = zq(18 % 17);
        let mut rng = rand::rng();
        let mut prover = SingleProductProver::new(table.clone(), table);
        let out = sumcheck(&mut prover, claimed, 2, 3, &mut rng);
        assert_eq!(out.rands.len(), 2);
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
        let ys = vec![zq(5), zq(7)];
        assert_eq!(evaluate_from_points(&ys, zq(0)), zq(5));
        assert_eq!(evaluate_from_points(&ys, zq(1)), zq(7));
    }

    #[test]
    fn test_interp_passes_through_given_points_3pt() {
        let ys = vec![zq(3), zq(5), zq(11)];
        assert_eq!(evaluate_from_points(&ys, zq(0)), zq(3));
        assert_eq!(evaluate_from_points(&ys, zq(1)), zq(5));
        assert_eq!(evaluate_from_points(&ys, zq(2)), zq(11));
    }

    #[test]
    fn test_interp_linear_eval_at_new_point() {
        let ys = ys_of(2, |x| (2 * x + 3) % Q);
        assert_eq!(evaluate_from_points(&ys, zq(5)), zq(13));
        assert_eq!(evaluate_from_points(&ys, zq(7)), zq(0));
    }

    #[test]
    fn test_interp_quadratic_eval_at_new_point() {
        let ys = ys_of(3, |x| (x * x + 1) % Q);
        assert_eq!(evaluate_from_points(&ys, zq(3)), zq(10));
        assert_eq!(evaluate_from_points(&ys, zq(4)), zq(0));
    }

    #[test]
    fn test_interp_quadratic_nontrivial_y0() {
        let ys = vec![zq(4), zq(9), zq(3)];
        assert_eq!(evaluate_from_points(&ys, zq(5)), zq(4));
    }

    #[test]
    fn test_interp_at_nonintegral_eval_point() {
        let ys = ys_of(2, |x| (x + 1) % Q);
        assert_eq!(evaluate_from_points(&ys, zq(16)), zq(0));
    }

    #[test]
    fn test_interp_4pt_cubic() {
        let ys = ys_of(4, |x| (x * x * x) % Q);
        assert_eq!(evaluate_from_points(&ys, zq(4)), zq(13));
        assert_eq!(evaluate_from_points(&ys, zq(2)), zq(8));
        assert_eq!(evaluate_from_points(&ys, zq(3)), zq(10));
    }

    #[test]
    fn test_interp_constant_polynomial() {
        let ys = vec![zq(7), zq(7), zq(7)];
        assert_eq!(evaluate_from_points(&ys, zq(0)), zq(7));
        assert_eq!(evaluate_from_points(&ys, zq(5)), zq(7));
        assert_eq!(evaluate_from_points(&ys, zq(13)), zq(7));
    }
}
