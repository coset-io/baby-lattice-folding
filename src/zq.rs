use std::ops::{Add, Mul, Neg, Sub};

/// Element of Z_q = integers mod q.
///
/// Invariant: `value` is always in [0, Q).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Zq<const Q: u64> {
    value: u64,
}

impl<const Q: u64> Zq<Q> {
    /// Create a new Zq element, reducing mod Q.
    pub fn new(value: u64) -> Self {
        Zq { value: value % Q }
    }

    /// Returns the inner value in [0, Q).
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Additive identity: 0 mod Q.
    pub fn zero() -> Self {
        Zq { value: 0 }
    }

    /// Multiplicative identity: 1 mod Q.
    pub fn one() -> Self {
        Zq { value: 1 }
    }
}

impl<const Q: u64> Add for Zq<Q> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        // TODO: overflow check
        Zq {
            value: (self.value + rhs.value) % Q,
        }
    }
}

impl<const Q: u64> Sub for Zq<Q> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        // a - b = (a + (q - b)) % b?
        if self.value >= rhs.value {
            Zq {
                value: self.value - rhs.value,
            }
        } else {
            // a < b -> a - b < 0 -> a + Q - b < Q
            Zq {
                value: self.value + Q - rhs.value,
            }
        }
    }
}

impl<const Q: u64> Mul for Zq<Q> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // TODO: overflow check
        Zq {
            value: (self.value * rhs.value) % Q,
        }
    }
}

impl<const Q: u64> Neg for Zq<Q> {
    type Output = Self;

    /// Additive inverse: -self mod Q.
    fn neg(self) -> Self {
        if self.value == 0 {
            self
        } else {
            Zq {
                value: Q - self.value,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Dev params: q = 17
    const Q: u64 = 17;

    type F = Zq<Q>;

    #[test]
    fn test_new_reduces() {
        assert_eq!(F::new(20).value(), 3); // 20 mod 17 = 3
        assert_eq!(F::new(Q).value(), 0);
        assert_eq!(F::new(0).value(), 0);
    }

    #[test]
    fn test_add() {
        assert_eq!(F::new(10) + F::new(5), F::new(15));
        assert_eq!(F::new(10).add(F::new(7)).value(), 0); // 17 mod 17
        assert_eq!(F::new(10).add(F::new(10)).value(), 3); // 20 mod 17
    }

    #[test]
    fn test_sub() {
        assert_eq!(F::new(10) - F::new(5), F::new(5));
        assert_eq!(F::new(5).sub(F::new(10)).value(), 12); // -5 mod 17 = 12
        assert_eq!(F::new(0).sub(F::new(1)).value(), 16); // -1 mod 17
    }

    #[test]
    fn test_mul() {
        assert_eq!(F::new(3) * F::new(5), F::new(15));
        assert_eq!(F::new(3).mul(F::new(6)).value(), 1); // 18 mod 17
        assert_eq!(F::new(0).mul(F::new(10)).value(), 0);
    }

    #[test]
    fn test_neg() {
        assert_eq!(-F::new(5), F::new(12)); // -5 mod 17
        assert_eq!(F::new(0).neg().value(), 0);
        assert_eq!(F::new(1).neg().value(), 16);
    }

    #[test]
    fn test_identities() {
        assert_eq!(F::zero().value(), 0);
        assert_eq!(F::one().value(), 1);
    }
}
