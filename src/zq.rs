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

    /// Modular exponentiation: self^exp mod Q (square-and-multiply).
    pub fn pow(self, exp: u64) -> Self {
        if exp == 0 {
            Self::one()
        } else {
            // f(x) = g^x
            //      = g*f(x/2)**2 if x \in odd else f(x/2)**2
            let half = self.pow(exp / 2);
            if exp.is_multiple_of(2) {
                half * half
            } else {
                self * (half * half)
            }
        }
    }

    /// Multiplicative inverse of v, i.e. s = v^{-1} s.t. v*s = 1 mod q
    pub fn inv(self) -> Self {
        assert!(self.value != 0, "cannout invert zero");
        // find s, t. s.t. sq + tv = 1
        let mut a = Q as i128;
        let mut b = self.value as i128;
        // q's multiplier
        let mut s = [1, 0] as [i128; 2];
        // v's multiplier
        let mut t = [0, 1] as [i128; 2];

        // gcd: (a, b) = (b, r) until b = 1
        while b > 1 {
            // a > b, b times a factor k and a minus bk
            let q = a / b;
            let r = a - q * b;
            let r_mplr = [s[0] - q * t[0], s[1] - q * t[1]];
            // (a, b) = (b, r)
            (a, b) = (b, r);
            (s, t) = (t, r_mplr);
        }

        // when b = 1, t[1] is v's multiplier, i.e. v^{-1}
        // Add Q to t[1] to ensure it's positive, then cast it back to u64
        // should be safe since we originally operate in u64.
        // mod Q again to enusre it's in range [0, Q)
        Zq {
            value: ((t[1] + (Q as i128)) as u64) % Q,
        }
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

    #[test]
    fn test_pow() {
        // 3^0 = 1
        assert_eq!(F::new(3).pow(0), F::one());
        // 3^1 = 3
        assert_eq!(F::new(3).pow(1), F::new(3));
        // 3^2 = 9
        assert_eq!(F::new(3).pow(2), F::new(9));
        // 3^3 = 27 mod 17 = 10
        assert_eq!(F::new(3).pow(3), F::new(10));
        // Fermat's little theorem: a^(q-1) = 1 for a != 0
        for i in 1..Q {
            assert_eq!(F::new(i).pow(Q - 1), F::one());
        }
        // 0^n = 0 for n > 0
        assert_eq!(F::new(0).pow(5), F::zero());
    }

    #[test]
    fn test_inv() {
        for i in 1..Q {
            let e = F::new(i);
            assert_eq!((e * e.inv()).value(), 1);
        }
    }

    #[test]
    #[should_panic]
    fn test_inv_zero() {
        F::new(0).inv();
    }
}
