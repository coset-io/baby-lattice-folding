use rand::Rng;
use std::ops::{Add, Mul, Neg, Sub};

use crate::ntt;
use crate::zq::Zq;

pub trait Ring:
    Sized
    + Copy
    + Clone
    + PartialEq
    + Eq
    + std::fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
}

/// Polynomial ring element in R_q = Z_q[X]/(X^d + 1).
///
/// Always has exactly D coefficients (fixed size).
/// Arithmetic automatically reduces mod X^D + 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    pub fn random(rng: &mut impl Rng) -> Self {
        Rq {
            coeffs: std::array::from_fn(|_| Zq::random(rng)),
        }
    }

    /// Convert to NTT (evaluation) form.
    pub fn ntt(self) -> RqNtt<Q, D> {
        let evals_vec = ntt::ntt::<Q, D>(self.coeffs.to_vec());
        RqNtt {
            evals: evals_vec.try_into().unwrap(),
        }
    }

    /// Get |r|_2^2
    pub fn l2_norm_squared(&self) -> u64 {
        self.coeffs
            .iter()
            .map(|&z| {
                let a = z.to_centered();
                (a * a) as u64
            })
            .sum()
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

impl<const Q: u64, const D: usize> Ring for Rq<Q, D> {
    fn zero() -> Self {
        Self::zero()
    }
    fn one() -> Self {
        Self::one()
    }
}

/// Polynomial ring element in R_q = Z_q[X]/(X^d + 1), but in evaluation form!
///
/// Always has exactly D evaluations of the roots (fixed size).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    pub fn random(rng: &mut impl Rng) -> Self {
        RqNtt {
            evals: std::array::from_fn(|_| Zq::random(rng)),
        }
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

impl<const Q: u64, const D: usize> Ring for RqNtt<Q, D> {
    fn zero() -> Self {
        Self::zero()
    }
    fn one() -> Self {
        Self::one()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: u64 = 17;
    const D: usize = 4;
    type F = Zq<Q>;
    type Ring4 = Rq<Q, D>;
    type Ntt = RqNtt<Q, D>;

    // helper: build ring poly from raw u64 coefficients
    fn rp(c: [u64; D]) -> Ring4 {
        Ring4::new(c.map(|v| F::new(v)))
    }

    fn ntt_from(c: [u64; D]) -> Ntt {
        Ntt::new(c.map(|v| F::new(v)))
    }

    // ─── Rq tests: R_q = Z_q[X]/(X^4 + 1), q=17, d=4 ───

    #[test]
    fn test_ring_zero_one() {
        assert_eq!(Ring4::zero(), rp([0, 0, 0, 0]));
        assert_eq!(Ring4::one(), rp([1, 0, 0, 0]));
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
        assert_eq!(a.clone() + (-a), Ring4::zero());
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
        assert_eq!(rp([3, 5, 7, 11]) * Ring4::zero(), Ring4::zero());
    }

    #[test]
    fn test_ring_mul_by_one() {
        assert_eq!(rp([3, 5, 7, 11]) * Ring4::one(), rp([3, 5, 7, 11]));
    }

    // ─── RqNtt tests: pointwise ops in evaluation form ───

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
        let a = Ring4::one();
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
        let one = Ring4::one();
        assert_eq!((a.clone().ntt() * one.ntt()).intt(), a);
    }
}
