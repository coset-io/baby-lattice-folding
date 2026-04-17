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
}
