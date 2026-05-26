use crate::{ring::Rq, zq::Zq};

/// [eq(x, (0,..., 0)), eq(x, (d-1, ..., d-1)]
pub fn tensor<const Q: u64, const D: usize>(x_vec: &[Rq<Q, D>], d_h: usize) -> Vec<Rq<Q, D>> {
    // (0..d_h).product()
    let l = x_vec.len();

    // Prepare for eqs_per_variate
    // [
    //   [eq(x_0, 0), ..., eq(x_0, d-1)],  // eq for variate x_0
    //   ...,
    //   [eq(x_{l-1}, 0), ..., eq(x_{l-1}, d-1)],  // eq for variate x_{l-1}
    // ]
    let eqs_per_variate: Vec<Vec<Rq<Q, D>>> = (0..l)
        .map(|j| {
            let t: Vec<Rq<Q, D>> = (0..d_h)
                .map(|k| {
                    (0..d_h)
                        .filter(|&k_prime| k_prime != k)
                        .map(|k_prime| {
                            (x_vec[j] - Rq::from_u64(k_prime as u64))
                                * (Zq::new(k as u64) - Zq::new(k_prime as u64)).inv()
                        })
                        // multiply altogether
                        .fold(Rq::one(), |acc, v| acc * v)
                })
                .collect();
            t
        })
        .collect();

    // Calculate the eq(x, idx) on the entire hypercube
    // by reusing the tensor product in Mat.
    // e.g. idx = (0, ..., 0)
    //  eq(x, idx) = eq(x, (0,..., 0)) = eq(x_0, 0) * ... * eq(x_{l-1}, 0)
    // let eqs = eqs_per_variate
    //     .iter()
    //     .skip(1)
    //     .fold(eqs_per_variate[0].clone(), |acc, v| acc.tensor_product(v));
    eqs_per_variate
        .iter()
        .skip(1)
        .fold(eqs_per_variate[0].clone(), |acc, v| {
            let mut ret_vec = Vec::with_capacity(acc.len() * v.len());
            for s in acc.iter() {
                for t in v.iter() {
                    ret_vec.push(*s * *t);
                }
            }
            ret_vec
        })
}

// LDE[w](x) = \sum_{\vec z \in [d_h]^l} eq(z, x) * w(\vec z)
//           = w_0*eq(x, (0,..., 0)) + ... + w_{d^l-1}*eq(x, (d-1, ..., d-1)
//           = <w, eqs>
pub fn lde<const Q: u64, const D: usize>(
    w: &[Rq<Q, D>],
    d_h: usize,
    x_vec: &[Rq<Q, D>],
) -> Rq<Q, D> {
    let len_w = w.len();
    let l = x_vec.len();
    // w must have the size of `d_h^l`. Otherwise it must have been padded.
    assert_eq!(len_w, d_h.pow(l as u32));

    let eq_vec = tensor(x_vec, d_h);
    // LDE[w](x) = <w, tensor(x)>
    w.iter()
        .zip(eq_vec)
        .map(|(&w_i, eq_i)| w_i * eq_i)
        .fold(Rq::zero(), |acc, v| acc + v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zq::Zq;

    const Q: u64 = 17;
    const D: usize = 4;
    type R = Rq<Q, D>;

    /// Helper: Rq element from raw u64 coefficients.
    fn r(c: [u64; D]) -> R {
        R::new(c.map(Zq::<Q>::new))
    }

    /// Helper: constant polynomial (degree 0).
    fn rc(v: u64) -> R {
        R::from_u64(v)
    }

    /// Sum of an Rq slice (Rq doesn't impl Sum).
    fn sum_rq(xs: &[R]) -> R {
        xs.iter().copied().fold(R::zero(), |acc, v| acc + v)
    }

    // ═══════════════════════════════════════════════════════════
    // tensor: per-variate Lagrange basis · Kronecker over variables
    // ═══════════════════════════════════════════════════════════
    //
    // tensor(x, d_h)[idx] = Π_j ℓ_{idx_j}(x_j)
    //   where ℓ_k is the Lagrange basis on {0, 1, ..., d_h - 1}.
    // Index ordering: `idx_0 * d_h^{l-1} + idx_1 * d_h^{l-2} + ... + idx_{l-1}`
    // (lex, x_0 most-significant) — matches Mat::tensor_product semantics.

    // ─── Shape ───

    #[test]
    fn test_tensor_len_eq_d_h_pow_l() {
        // |tensor(x, d_h)| = d_h^l  for any x_vec of length l.
        for l in 1..=4 {
            for d_h in 2..=3 {
                let x_vec = vec![rc(7); l];
                let t = tensor(&x_vec, d_h);
                assert_eq!(t.len(), d_h.pow(l as u32), "l={l}, d_h={d_h}");
            }
        }
    }

    // ─── d_h = 2 (boolean hypercube) — bug-invisible cases ───

    #[test]
    fn test_tensor_d_h_2_l_1_at_zero() {
        // ℓ_0(0) = 1, ℓ_1(0) = 0.
        let t = tensor(&[rc(0)], 2);
        assert_eq!(t, vec![rc(1), rc(0)]);
    }

    #[test]
    fn test_tensor_d_h_2_l_1_at_one() {
        // ℓ_0(1) = 0, ℓ_1(1) = 1.
        let t = tensor(&[rc(1)], 2);
        assert_eq!(t, vec![rc(0), rc(1)]);
    }

    #[test]
    fn test_tensor_d_h_2_l_1_at_five() {
        // d_h=2: ℓ_0(x) = 1 - x, ℓ_1(x) = x.
        // x = 5: ℓ_0 = -4 = 13, ℓ_1 = 5.
        let t = tensor(&[rc(5)], 2);
        assert_eq!(t, vec![rc(13), rc(5)]);
    }

    #[test]
    fn test_tensor_d_h_2_l_2_delta_property() {
        // At hypercube point z = (z_0, z_1), tensor[z_0 * d_h + z_1] = 1, others 0.
        for z0 in 0..2 {
            for z1 in 0..2 {
                let t = tensor(&[rc(z0), rc(z1)], 2);
                let idx = (z0 as usize) * 2 + (z1 as usize);
                for (i, v) in t.iter().enumerate() {
                    let expected = if i == idx { R::one() } else { R::zero() };
                    assert_eq!(*v, expected, "z=({z0},{z1}), i={i}");
                }
            }
        }
    }

    #[test]
    fn test_tensor_d_h_2_l_2_at_non_hypercube_point() {
        // x = (2, 3) over Z_17.
        // ℓ_0(2)·ℓ_0(3) = (1-2)(1-3) = (-1)·(-2) = 2
        // ℓ_0(2)·ℓ_1(3) = (1-2)·3     = -3 = 14
        // ℓ_1(2)·ℓ_0(3) =  2·(1-3)    = -4 = 13
        // ℓ_1(2)·ℓ_1(3) =  2·3        = 6
        let t = tensor(&[rc(2), rc(3)], 2);
        assert_eq!(t, vec![rc(2), rc(14), rc(13), rc(6)]);
    }

    #[test]
    fn test_tensor_d_h_2_l_1_at_nonconst_rq() {
        // x = X (indeterminate).  ℓ_0(X) = 1 - X,  ℓ_1(X) = X.
        let t = tensor(&[r([0, 1, 0, 0])], 2);
        assert_eq!(t, vec![r([1, 16, 0, 0]), r([0, 1, 0, 0])]);
    }

    // ─── d_h = 3 — bug-visible cases (the (k - k') = ±2 denominator) ───

    #[test]
    fn test_tensor_d_h_3_l_1_at_hypercube_points() {
        // ℓ_k(k') = δ_{k,k'} on hypercube {0, 1, 2}.
        for z in 0..3u64 {
            let t = tensor(&[rc(z)], 3);
            let mut expected = vec![rc(0); 3];
            expected[z as usize] = rc(1);
            assert_eq!(t, expected, "delta property fails at z={z} (d_h=3)");
        }
    }

    #[test]
    fn test_tensor_d_h_3_l_1_at_three() {
        // Hand computed: ℓ_0(3) = (3-1)(3-2)/((0-1)(0-2)) = 2·1/2 = 1
        //                ℓ_1(3) = (3-0)(3-2)/((1-0)(1-2)) = 3·1/(-1) = -3 = 14
        //                ℓ_2(3) = (3-0)(3-1)/((2-0)(2-1)) = 3·2/2 = 3
        let t = tensor(&[rc(3)], 3);
        assert_eq!(t, vec![rc(1), rc(14), rc(3)]);
    }

    #[test]
    fn test_tensor_d_h_3_l_2_delta_at_corners() {
        // 9 hypercube corners over {0,1,2}^2 (d_h=3).
        for z0 in 0..3u64 {
            for z1 in 0..3u64 {
                let t = tensor(&[rc(z0), rc(z1)], 3);
                let idx = (z0 as usize) * 3 + (z1 as usize);
                for (i, v) in t.iter().enumerate() {
                    let expected = if i == idx { R::one() } else { R::zero() };
                    assert_eq!(*v, expected, "z=({z0},{z1}), i={i}, d_h=3");
                }
            }
        }
    }

    // ─── Partition of unity: Σ_i tensor[i] ≡ 1 for any x ───
    //
    // This is a defining property of any Lagrange-basis tensor: the basis
    // partitions unity at every point, not just on the hypercube. It catches
    // the same denominator bug at d_h=3 even when query point is "generic".

    #[test]
    fn test_tensor_partition_of_unity_d_h_2() {
        // Try a handful of x's including non-hypercube ones.
        for x0 in 0..5u64 {
            for x1 in 0..5u64 {
                let t = tensor(&[rc(x0), rc(x1)], 2);
                assert_eq!(sum_rq(&t), R::one(), "x=({x0},{x1}), d_h=2");
            }
        }
    }

    #[test]
    fn test_tensor_partition_of_unity_d_h_3() {
        for x in 0..5u64 {
            let t = tensor(&[rc(x)], 3);
            assert_eq!(sum_rq(&t), R::one(), "x={x}, d_h=3");
        }
        for x0 in 0..3u64 {
            for x1 in 0..3u64 {
                let t = tensor(&[rc(x0), rc(x1)], 3);
                assert_eq!(sum_rq(&t), R::one(), "x=({x0},{x1}), d_h=3");
            }
        }
    }

    #[test]
    fn test_tensor_partition_of_unity_nonconst_rq() {
        // Even with a non-constant Rq evaluation point, the partition holds.
        let t = tensor(&[r([0, 1, 0, 0]), r([1, 1, 0, 0])], 2); // x_0=X, x_1=1+X
        assert_eq!(sum_rq(&t), R::one());
    }

    // ═══════════════════════════════════════════════════════════
    // lde: multilinear/multi-degree-(d_h-1) extension of w
    // ═══════════════════════════════════════════════════════════
    //
    // LDE[w](x) = <w, tensor(x, d_h)>.  Defining properties:
    //   (1) Agreement on hypercube: LDE[w](z) = w[idx(z)] for z ∈ [d_h]^l.
    //   (2) Linearity in w: LDE[α·w + β·w'] = α·LDE[w] + β·LDE[w'].
    //   (3) Constant: LDE[(c,...,c)](x) = c  (partition-of-unity corollary).

    // ─── d_h = 2: hypercube agreement ───

    #[test]
    fn test_lde_d_h_2_l_2_agrees_on_hypercube() {
        // w[z_0 * 2 + z_1] at corner z = (z_0, z_1).
        let w = vec![rc(7), rc(11), rc(5), rc(3)];
        for z0 in 0..2 {
            for z1 in 0..2 {
                let v = lde(&w, 2, &[rc(z0), rc(z1)]);
                let idx = (z0 as usize) * 2 + (z1 as usize);
                assert_eq!(v, w[idx], "fail at z=({z0},{z1})");
            }
        }
    }

    #[test]
    fn test_lde_d_h_2_l_2_at_non_hypercube() {
        // w = [1, 2, 3, 4] (w_00=1, w_01=2, w_10=3, w_11=4).
        // The multilinear extension expands to: 1 + 2 x_0 + x_1 + 0·x_0·x_1.
        // At (x_0, x_1) = (2, 3) over Z_17: 1 + 4 + 3 = 8.
        let w = vec![rc(1), rc(2), rc(3), rc(4)];
        assert_eq!(lde(&w, 2, &[rc(2), rc(3)]), rc(8));
    }

    #[test]
    fn test_lde_d_h_2_l_3_agrees_on_hypercube() {
        // 8-point cube; check all corners against the bookkeeping order
        //   idx = z_0 * 4 + z_1 * 2 + z_2.
        let w: Vec<R> = (0..8u64).map(rc).collect();
        for z0 in 0..2 {
            for z1 in 0..2 {
                for z2 in 0..2 {
                    let v = lde(&w, 2, &[rc(z0), rc(z1), rc(z2)]);
                    let idx = (z0 as usize) * 4 + (z1 as usize) * 2 + z2 as usize;
                    assert_eq!(v, w[idx], "fail at z=({z0},{z1},{z2})");
                }
            }
        }
    }

    // ─── d_h = 2: structural properties ───

    #[test]
    fn test_lde_zero_vector_is_zero_everywhere() {
        let w = vec![R::zero(); 4];
        // Any evaluation point.
        for (x0, x1) in [(0, 0), (1, 1), (3, 7), (16, 5)] {
            assert_eq!(lde(&w, 2, &[rc(x0), rc(x1)]), R::zero(), "x=({x0},{x1})");
        }
    }

    #[test]
    fn test_lde_constant_function_equals_constant() {
        // w = [c, c, c, c]  →  LDE[w] ≡ c (anywhere).
        let c = rc(9);
        let w = vec![c; 4];
        for (x0, x1) in [(0, 0), (1, 0), (3, 7), (16, 5)] {
            assert_eq!(lde(&w, 2, &[rc(x0), rc(x1)]), c, "x=({x0},{x1})");
        }
    }

    #[test]
    fn test_lde_linearity_in_w() {
        // LDE[α·w + β·w'](x) == α·LDE[w](x) + β·LDE[w'](x).
        let w = vec![rc(1), rc(2), rc(3), rc(4)];
        let w_prime = vec![rc(5), rc(6), rc(7), rc(8)];
        let alpha = rc(2);
        let beta = rc(3);
        let combined: Vec<R> = w
            .iter()
            .zip(&w_prime)
            .map(|(&a, &b)| alpha * a + beta * b)
            .collect();
        let x = &[rc(4), rc(5)];
        let lhs = lde(&combined, 2, x);
        let rhs = alpha * lde(&w, 2, x) + beta * lde(&w_prime, 2, x);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_lde_at_nonconst_rq_evaluation_point() {
        // w = [1, 2, 3, 4]  →  multilinear = 1 + 2 x_0 + x_1.
        // x_0 = X (indeterminate), x_1 = 1 - X:
        //   LDE = 1 + 2 X + (1 - X) = 2 + X = [2, 1, 0, 0].
        let w = vec![rc(1), rc(2), rc(3), rc(4)];
        let x = &[r([0, 1, 0, 0]), r([1, 16, 0, 0])]; // [X, 1-X]
        assert_eq!(lde(&w, 2, x), r([2, 1, 0, 0]));
    }

    #[test]
    fn test_lde_e_i_equals_basis_function() {
        // w = e_i (1 at position i, 0 elsewhere) → LDE[e_i](x) = tensor(x)[i].
        // Verifies <w, tensor(x)> machinery picks out the right basis function.
        let x = &[rc(3), rc(5)];
        let t = tensor(x, 2);
        for i in 0..4 {
            let mut w = vec![R::zero(); 4];
            w[i] = R::one();
            assert_eq!(lde(&w, 2, x), t[i], "fail at e_{i}");
        }
    }

    #[test]
    #[should_panic]
    fn test_lde_panics_on_size_mismatch() {
        // w.len() must equal d_h^l.  Pass len=3 with d_h=2, l=2 (expects 4).
        let w = vec![rc(1), rc(2), rc(3)];
        let _ = lde(&w, 2, &[rc(0), rc(0)]);
    }

    // ─── d_h = 3 — bug-visible cases ───

    #[test]
    fn test_lde_d_h_3_l_1_agrees_on_hypercube() {
        // w = [3, 5, 11] at x ∈ {0, 1, 2}.
        let w = vec![rc(3), rc(5), rc(11)];
        assert_eq!(lde(&w, 3, &[rc(0)]), rc(3));
        assert_eq!(lde(&w, 3, &[rc(1)]), rc(5));
        assert_eq!(lde(&w, 3, &[rc(2)]), rc(11));
    }

    #[test]
    fn test_lde_d_h_3_l_1_at_x_equals_three() {
        // p(x) = 2 x^2 + 3 over Z_17:  p(0)=3, p(1)=5, p(2)=11, p(3) = 21 = 4.
        let w = vec![rc(3), rc(5), rc(11)];
        assert_eq!(lde(&w, 3, &[rc(3)]), rc(4));
    }

    #[test]
    fn test_lde_d_h_3_l_1_constant_function() {
        let w = vec![rc(7); 3];
        for x in [0, 1, 2, 5, 13] {
            assert_eq!(lde(&w, 3, &[rc(x)]), rc(7), "x={x}, d_h=3");
        }
    }

    #[test]
    fn test_lde_d_h_3_l_2_agrees_on_hypercube() {
        // 9-point square; idx = z_0 * 3 + z_1.
        let w: Vec<R> = (0..9u64).map(rc).collect();
        for z0 in 0..3u64 {
            for z1 in 0..3u64 {
                let v = lde(&w, 3, &[rc(z0), rc(z1)]);
                let idx = (z0 as usize) * 3 + (z1 as usize);
                assert_eq!(v, w[idx], "fail at z=({z0},{z1}), d_h=3");
            }
        }
    }
}
