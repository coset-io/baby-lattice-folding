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
}
