use super::ntt;
use super::zq::Zq;
use std::ops::{Add, Mul, Neg, Sub};

/// Polynomial
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Poly<const Q: u64> {
    coeffs: Vec<Zq<Q>>,
}

impl<const Q: u64> Poly<Q> {
    pub fn new(coeffs: impl Into<Vec<Zq<Q>>>) -> Self {
        Poly {
            coeffs: coeffs.into(),
        }
    }

    pub fn zero() -> Self {
        Poly { coeffs: Vec::new() }
    }

    pub fn one() -> Self {
        Poly {
            coeffs: vec![Zq::one()],
        }
    }

    pub fn eval(&self, x: u64) -> Zq<Q> {
        let mut s = Zq::<Q>::zero();
        for (i, &c) in self.coeffs.iter().enumerate() {
            s = s + c * Zq::<Q>::new(x).pow(i as u64);
        }
        s
    }
}

impl<const Q: u64> Add for Poly<Q> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let len = self.coeffs.len().max(rhs.coeffs.len());
        let mut new_coeffs = vec![Zq::<Q>::zero(); len];
        for (i, &c) in self.coeffs.iter().enumerate() {
            new_coeffs[i] = new_coeffs[i] + c;
        }
        for (i, &c) in rhs.coeffs.iter().enumerate() {
            new_coeffs[i] = new_coeffs[i] + c;
        }

        // trim 0s in the top terms
        while new_coeffs.last() == Some(&Zq::zero()) {
            new_coeffs.pop();
        }
        Poly { coeffs: new_coeffs }
    }
}

impl<const Q: u64> Neg for Poly<Q> {
    type Output = Self;

    fn neg(self) -> Self {
        Poly {
            coeffs: self.coeffs.iter().map(|&c| -c).collect(),
        }
    }
}

impl<const Q: u64> Sub for Poly<Q> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<const Q: u64> Mul for Poly<Q> {
    type Output = Self;

    #[allow(clippy::needless_range_loop)]
    fn mul(self, rhs: Self) -> Self {
        if self.coeffs.is_empty() || rhs.coeffs.is_empty() {
            return Poly::zero();
        }

        let new_len = self.coeffs.len() + rhs.coeffs.len() - 1;
        let mut new_coeffs = vec![Zq::<Q>::zero(); new_len];
        for i in 0..self.coeffs.len() {
            for j in 0..rhs.coeffs.len() {
                new_coeffs[i + j] = new_coeffs[i + j] + self.coeffs[i] * rhs.coeffs[j]
            }
        }

        // trim 0s in the top terms
        while new_coeffs.last() == Some(&Zq::zero()) {
            new_coeffs.pop();
        }

        Poly { coeffs: new_coeffs }
    }
}

/// Polynomial ring element in R_q = Z_q[X]/(X^d + 1).
///
/// Always has exactly D coefficients (fixed size).
/// Arithmetic automatically reduces mod X^D + 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rq<const Q: u64, const D: usize> {
    coeffs: [Zq<Q>; D],
}

impl<const Q: u64, const D: usize> Rq<Q, D> {
    pub fn new(coeffs: [Zq<Q>; D]) -> Self {
        Rq { coeffs }
    }

    pub fn zero() -> Self {
        Rq {
            coeffs: [Zq::zero(); D],
        }
    }

    pub fn one() -> Self {
        let mut coeffs = [Zq::zero(); D];
        coeffs[0] = Zq::one();
        Rq { coeffs }
    }

    pub fn coeffs(&self) -> &[Zq<Q>; D] {
        &self.coeffs
    }

    /// Convert to NTT (evaluation) form.
    pub fn ntt(self) -> RqNtt<Q, D> {
        let evals_vec = ntt::ntt::<Q, D>(self.coeffs.to_vec());
        RqNtt {
            evals: evals_vec.try_into().unwrap(),
        }
    }

    /// Reduce a polynomial (with up to 2D-1 coefficients) mod X^D + 1.
    fn reduce(full: &[Zq<Q>]) -> [Zq<Q>; D] {
        assert!(full.len() < 2 * D);
        let mut coeffs = [Zq::zero(); D];
        for (i, &c) in full.iter().enumerate() {
            if i < D {
                coeffs[i] = coeffs[i] + c;
            } else {
                // X^{D+i} = -X^i, subtract
                coeffs[i - D] = coeffs[i - D] - c;
            }
        }
        coeffs
    }
}

impl<const Q: u64, const D: usize> Add for Rq<Q, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Rq {
            coeffs: std::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]),
        }
    }
}

impl<const Q: u64, const D: usize> Neg for Rq<Q, D> {
    type Output = Self;

    fn neg(self) -> Self {
        Rq {
            coeffs: self.coeffs.map(|c| -c),
        }
    }
}

impl<const Q: u64, const D: usize> Sub for Rq<Q, D> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<const Q: u64, const D: usize> Mul for Rq<Q, D> {
    type Output = Self;

    #[allow(clippy::needless_range_loop)]
    fn mul(self, rhs: Self) -> Self {
        let mut new_coeffs = vec![Zq::<Q>::zero(); self.coeffs.len() + rhs.coeffs.len() - 1];
        for i in 0..self.coeffs.len() {
            for j in 0..rhs.coeffs.len() {
                new_coeffs[i + j] = new_coeffs[i + j] + self.coeffs[i] * rhs.coeffs[j];
            }
        }
        Rq {
            coeffs: Self::reduce(&new_coeffs),
        }
    }
}

/// Polynomial ring element in R_q = Z_q[X]/(X^d + 1), but in evaluation form!
///
/// Always has exactly D evaluations of the roots (fixed size).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RqNtt<const Q: u64, const D: usize> {
    evals: [Zq<Q>; D],
}

impl<const Q: u64, const D: usize> RqNtt<Q, D> {
    pub fn new(evals: [Zq<Q>; D]) -> Self {
        RqNtt { evals }
    }

    pub fn zero() -> Self {
        RqNtt {
            evals: [Zq::zero(); D],
        }
    }

    pub fn one() -> Self {
        RqNtt {
            evals: [Zq::one(); D],
        }
    }

    pub fn evals(&self) -> &[Zq<Q>; D] {
        &self.evals
    }

    /// Convert back to coefficient form.
    pub fn intt(self) -> Rq<Q, D> {
        let coeffs_vec = ntt::intt::<Q, D>(self.evals.to_vec());
        Rq {
            coeffs: coeffs_vec.try_into().unwrap(),
        }
    }
}

impl<const Q: u64, const D: usize> Add for RqNtt<Q, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        RqNtt {
            evals: std::array::from_fn(|i| self.evals[i] + rhs.evals[i]),
        }
    }
}

impl<const Q: u64, const D: usize> Neg for RqNtt<Q, D> {
    type Output = Self;

    fn neg(self) -> Self {
        RqNtt {
            evals: self.evals.map(|c| -c),
        }
    }
}

impl<const Q: u64, const D: usize> Sub for RqNtt<Q, D> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<const Q: u64, const D: usize> Mul for RqNtt<Q, D> {
    type Output = Self;

    #[allow(clippy::needless_range_loop)]
    fn mul(self, rhs: Self) -> Self {
        RqNtt {
            evals: std::array::from_fn(|i| self.evals[i] * rhs.evals[i]),
        }
    }
}

#[cfg(test)]
mod tests {
    // import everything in this module
    use super::*;

    const Q: u64 = 17;
    type F = Zq<Q>;
    type R = Poly<Q>;

    // helper: build poly from raw u64 coefficients
    fn p(coeffs: &[u64]) -> R {
        R::new(coeffs.iter().map(|&c| F::new(c)).collect::<Vec<_>>())
    }

    #[test]
    fn test_new() {
        assert_eq!(R::new(vec![]), R::zero());
    }

    #[test]
    fn test_one() {
        assert_eq!(R::one(), p(&[1]));
    }

    #[test]
    fn test_add_same_len() {
        // (3 + 5x) + (2 + 4x) = 5 + 9x
        assert_eq!(p(&[3, 5]) + p(&[2, 4]), p(&[5, 9]));
    }

    #[test]
    fn test_add_different_len() {
        // (1 + 2x + 3x^2) + (4 + 5x) = 5 + 7x + 3x^2
        assert_eq!(p(&[1, 2, 3]) + p(&[4, 5]), p(&[5, 7, 3]));
    }

    #[test]
    fn test_add_with_cancellation() {
        // (1 + 16x) + (2 + x) = 3 + 0x = 3  (16 + 1 = 17 ≡ 0 mod 17)
        // trailing zero should be trimmed
        assert_eq!(p(&[1, 16]) + p(&[2, 1]), p(&[3]));
    }

    #[test]
    fn test_add_zero() {
        assert_eq!(p(&[3, 5]) + R::zero(), p(&[3, 5]));
        assert_eq!(R::zero() + p(&[3, 5]), p(&[3, 5]));
    }

    #[test]
    fn test_neg() {
        // -(3 + 5x) = 14 + 12x  (mod 17)
        assert_eq!(-p(&[3, 5]), p(&[14, 12]));
    }

    #[test]
    fn test_neg_zero() {
        assert_eq!(-R::zero(), R::zero());
    }

    #[test]
    fn test_sub() {
        // (10 + 3x) - (5 + 7x) = 5 + (-4 mod 17)x = 5 + 13x
        assert_eq!(p(&[10, 3]) - p(&[5, 7]), p(&[5, 13]));
    }

    #[test]
    fn test_add_sub_inverse() {
        let a = p(&[3, 5, 7]);
        assert_eq!(a.clone() + (-a), R::zero());
    }

    #[test]
    fn test_mul_basic() {
        // (1 + 2x) * (3 + 4x) = 3 + 4x + 6x + 8x^2 = 3 + 10x + 8x^2
        assert_eq!(p(&[1, 2]) * p(&[3, 4]), p(&[3, 10, 8]));
    }

    #[test]
    fn test_mul_with_mod() {
        // (5 + 4x) * (4 + 3x) = 20 + 15x + 16x + 12x^2
        //                      = 3 + 14x + 12x^2  (mod 17)
        assert_eq!(p(&[5, 4]) * p(&[4, 3]), p(&[3, 14, 12]));
    }

    #[test]
    fn test_mul_by_zero() {
        assert_eq!(p(&[3, 5]) * R::zero(), R::zero());
        assert_eq!(R::zero() * R::zero(), R::zero());
    }

    #[test]
    fn test_mul_by_one() {
        assert_eq!(p(&[3, 5, 7]) * R::one(), p(&[3, 5, 7]));
    }

    #[test]
    fn test_mul_by_constant() {
        // (1 + 2x + 3x^2) * (2) = 2 + 4x + 6x^2
        assert_eq!(p(&[1, 2, 3]) * p(&[2]), p(&[2, 4, 6]));
    }

    // ─── Poly::eval tests ───

    #[test]
    fn test_eval_constant() {
        // f = 5, f(x) = 5 for all x
        assert_eq!(p(&[5]).eval(0), F::new(5));
        assert_eq!(p(&[5]).eval(3), F::new(5));
    }

    #[test]
    fn test_eval_linear() {
        // f = 3 + 5x
        // f(0) = 3, f(1) = 8, f(2) = 13
        assert_eq!(p(&[3, 5]).eval(0), F::new(3));
        assert_eq!(p(&[3, 5]).eval(1), F::new(8));
        assert_eq!(p(&[3, 5]).eval(2), F::new(13));
    }

    #[test]
    fn test_eval_quadratic() {
        // f = 3 + 5x + 2x^2  (mod 17)
        // f(0) = 3
        // f(1) = 3 + 5 + 2 = 10
        // f(2) = 3 + 10 + 8 = 21 mod 17 = 4
        // f(16) = f(-1 mod 17) = 3 - 5 + 2 = 0
        assert_eq!(p(&[3, 5, 2]).eval(0), F::new(3));
        assert_eq!(p(&[3, 5, 2]).eval(1), F::new(10));
        assert_eq!(p(&[3, 5, 2]).eval(2), F::new(4));
        assert_eq!(p(&[3, 5, 2]).eval(16), F::new(0));
    }

    #[test]
    fn test_eval_zero_poly() {
        assert_eq!(R::zero().eval(5), F::zero());
    }

    // ─── Rq tests: R_q = Z_q[X]/(X^4 + 1), q=17, d=4 ───

    const D: usize = 4;
    type Ring = Rq<Q, D>;

    // helper: build ring poly from raw u64 coefficients
    fn rp(c: [u64; D]) -> Ring {
        Ring::new(c.map(|v| F::new(v)))
    }

    #[test]
    fn test_ring_zero_one() {
        assert_eq!(Ring::zero(), rp([0, 0, 0, 0]));
        assert_eq!(Ring::one(), rp([1, 0, 0, 0]));
    }

    #[test]
    fn test_ring_add() {
        // (1 + 2x + 3x^2 + 4x^3) + (5 + 6x + 7x^2 + 8x^3)
        // = 6 + 8x + 10x^2 + 12x^3
        assert_eq!(rp([1, 2, 3, 4]) + rp([5, 6, 7, 8]), rp([6, 8, 10, 12]));
    }

    #[test]
    fn test_ring_add_with_mod() {
        // (10 + 0x + 0x^2 + 0x^3) + (10 + 0x + 0x^2 + 0x^3)
        // = 20 mod 17 = 3
        assert_eq!(rp([10, 0, 0, 0]) + rp([10, 0, 0, 0]), rp([3, 0, 0, 0]));
    }

    #[test]
    fn test_ring_neg() {
        // -(1 + 2x + 3x^2 + 4x^3) = 16 + 15x + 14x^2 + 13x^3  (mod 17)
        assert_eq!(-rp([1, 2, 3, 4]), rp([16, 15, 14, 13]));
    }

    #[test]
    fn test_ring_sub() {
        assert_eq!(rp([5, 8, 3, 1]) - rp([2, 3, 1, 1]), rp([3, 5, 2, 0]));
    }

    #[test]
    fn test_ring_add_sub_inverse() {
        let a = rp([3, 5, 7, 11]);
        assert_eq!(a.clone() + (-a), Ring::zero());
    }

    #[test]
    fn test_ring_mul_no_reduction() {
        // (1 + x) * (1 + x) = 1 + 2x + x^2 (degree 2 < 4, no reduction needed)
        assert_eq!(rp([1, 1, 0, 0]) * rp([1, 1, 0, 0]), rp([1, 2, 1, 0]));
    }

    #[test]
    fn test_ring_mul_with_reduction() {
        // x^3 * x = x^4 ≡ -1 (mod X^4 + 1) = 16 (mod 17)
        assert_eq!(rp([0, 0, 0, 1]) * rp([0, 1, 0, 0]), rp([16, 0, 0, 0]));
    }

    #[test]
    fn test_ring_mul_full_reduction() {
        // (1 + x^3) * (1 + x^2) = 1 + x^2 + x^3 + x^5
        // x^5 = x^{4+1} = -x = 16x (mod 17)
        // result: 1 + 16x + x^2 + x^3
        assert_eq!(rp([1, 0, 0, 1]) * rp([1, 0, 1, 0]), rp([1, 16, 1, 1]));
    }

    #[test]
    fn test_ring_mul_by_zero() {
        assert_eq!(rp([3, 5, 7, 11]) * Ring::zero(), Ring::zero());
    }

    #[test]
    fn test_ring_mul_by_one() {
        assert_eq!(rp([3, 5, 7, 11]) * Ring::one(), rp([3, 5, 7, 11]));
    }

    // ─── RqNtt tests: pointwise ops in evaluation form ───

    type Ntt = RqNtt<Q, D>;

    fn ntt_from(c: [u64; D]) -> Ntt {
        Ntt::new(c.map(|v| F::new(v)))
    }

    #[test]
    fn test_ntt_zero_one() {
        // zero in eval form = all zeros
        assert_eq!(Ntt::zero(), ntt_from([0, 0, 0, 0]));
        // one in eval form = all ones (constant 1 evaluates to 1 everywhere)
        assert_eq!(Ntt::one(), ntt_from([1, 1, 1, 1]));
    }

    #[test]
    fn test_ntt_add() {
        // pointwise: [1,2,3,4] + [5,6,7,8] = [6,8,10,12]
        assert_eq!(
            ntt_from([1, 2, 3, 4]) + ntt_from([5, 6, 7, 8]),
            ntt_from([6, 8, 10, 12])
        );
    }

    #[test]
    fn test_ntt_add_with_mod() {
        // 10 + 10 = 20 mod 17 = 3
        assert_eq!(
            ntt_from([10, 10, 10, 10]) + ntt_from([10, 10, 10, 10]),
            ntt_from([3, 3, 3, 3])
        );
    }

    #[test]
    fn test_ntt_neg() {
        // -[1,2,3,4] = [16,15,14,13] (mod 17)
        assert_eq!(-ntt_from([1, 2, 3, 4]), ntt_from([16, 15, 14, 13]));
    }

    #[test]
    fn test_ntt_sub() {
        assert_eq!(
            ntt_from([10, 8, 5, 3]) - ntt_from([3, 2, 1, 1]),
            ntt_from([7, 6, 4, 2])
        );
    }

    #[test]
    fn test_ntt_add_sub_inverse() {
        let a = ntt_from([3, 5, 7, 11]);
        assert_eq!(a.clone() + (-a), Ntt::zero());
    }

    #[test]
    fn test_ntt_mul_pointwise() {
        // pointwise: [2,3,4,5] * [3,4,5,6] = [6,12,20,30]
        // mod 17: [6, 12, 3, 13]
        assert_eq!(
            ntt_from([2, 3, 4, 5]) * ntt_from([3, 4, 5, 6]),
            ntt_from([6, 12, 3, 13])
        );
    }

    #[test]
    fn test_ntt_mul_by_zero() {
        assert_eq!(ntt_from([3, 5, 7, 11]) * Ntt::zero(), Ntt::zero());
    }

    #[test]
    fn test_ntt_mul_by_one() {
        // mul by one = identity (pointwise * 1)
        let a = ntt_from([3, 5, 7, 11]);
        assert_eq!(a.clone() * Ntt::one(), a);
    }

    #[test]
    fn test_ntt_mul_self_inverse() {
        // a * inv(a) = 1 pointwise, for nonzero entries
        let a = ntt_from([2, 3, 5, 7]);
        let a_inv = Ntt::new(a.evals().map(|e| e.inv()));
        assert_eq!(a * a_inv, Ntt::one());
    }

    #[test]
    fn test_ntt_mul_commutativity() {
        let a = ntt_from([2, 5, 8, 11]);
        let b = ntt_from([3, 7, 13, 1]);
        assert_eq!(a.clone() * b.clone(), b * a);
    }

    #[test]
    fn test_ntt_mul_associativity() {
        let a = ntt_from([2, 5, 8, 11]);
        let b = ntt_from([3, 7, 13, 1]);
        let c = ntt_from([4, 9, 2, 6]);
        assert_eq!((a.clone() * b.clone()) * c.clone(), a * (b * c));
    }

    #[test]
    fn test_ntt_distributivity() {
        // a * (b + c) == a*b + a*c
        let a = ntt_from([2, 5, 8, 11]);
        let b = ntt_from([3, 7, 13, 1]);
        let c = ntt_from([4, 9, 2, 6]);
        assert_eq!(a.clone() * (b.clone() + c.clone()), a.clone() * b + a * c);
    }

    // ─── Rq <-> RqNtt conversion tests ───

    #[test]
    fn test_rq_ntt_roundtrip() {
        let a = rp([10, 4, 8, 0]);
        assert_eq!(a.clone().ntt().intt(), a);
    }

    #[test]
    fn test_rq_ntt_roundtrip_ones() {
        let a = Ring::one();
        assert_eq!(a.clone().ntt().intt(), a);
    }

    #[test]
    fn test_rq_ntt_mul_matches_schoolbook() {
        // NTT mul should give same result as schoolbook mul
        let a = rp([1, 0, 0, 1]); // 1 + x^3
        let b = rp([1, 0, 1, 0]); // 1 + x^2
        let schoolbook = a.clone() * b.clone();
        let ntt_result = (a.ntt() * b.ntt()).intt();
        assert_eq!(ntt_result, schoolbook);
    }

    #[test]
    fn test_rq_ntt_mul_by_one() {
        let a = rp([3, 5, 7, 11]);
        let one = Ring::one();
        assert_eq!((a.clone().ntt() * one.ntt()).intt(), a);
    }
}
