use core::panic;

use super::zq::Zq;
/*
src/
  poly.rs    ← Poly, Rq, RqNtt
  ntt.rs     ← ntt(), intt() methods

Rq <-> RqNTT with methods

impl Rq<Q, D> {
    fn ntt(self) -> RqNtt<Q, D> { ... }
}
impl RqNtt<Q, D> {
    fn intt(self) -> Rq<Q, D> { ... }
}
 */

pub fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut d = 2;
    while d * d <= n {
        if n.is_multiple_of(d) {
            factors.push(d);
            while n.is_multiple_of(d) {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

pub fn find_primitive_2d_root_of_unity<const Q: u64>(d: u64) -> Zq<Q> {
    // R_q = Z_q[X]/(X^d+1). X^d + 1 = 0 -> X^d = -1 \mod q
    // -> X^{2d} = 1. Assume q is prime, Z_q^* is a cyclic group with order q-1
    // i.e. \forall g \in Z_q^*, g^{q-1} = 1. Since g^{(q-1)/(2d)}^{2d} = 1,
    // for g to exist 2d must divide (q-1).
    let order = Q - 1;
    assert_eq!(
        order % (2 * d),
        0,
        "primitive {}-th root of unity doesn't exist (Q={})",
        2 * d,
        Q
    );
    // find a generator: if g is a generator, g^{q-1} = 1.
    // To make sure g didn't wrap around, i.e. g^{(q-1)/k} = 1, we just
    // test all prime factors of q-1 and make sure all of them satisfy
    // g^{(q-1)/k} != 1.
    let factors = prime_factors(order);
    for i in 2..order {
        let g = Zq::<Q>::new(i);
        // o(g) =
        let is_generator = factors.iter().all(|&p| g.pow(order / p) != Zq::<Q>::one());
        if is_generator {
            // if g is a generator -> o(g) = q-1 -> o(g^{(q-1)/2d}) = 2d
            return g.pow(order / (2 * d));
        }
    }
    panic!("no multiplicative generator found for Q={Q}")
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: u64 = 17;
    const D: u64 = 4;

    #[test]
    fn test_primitive_2d_root_of_unity() {
        let omega = find_primitive_2d_root_of_unity::<Q>(D);
        assert_eq!(omega.pow(2 * D), Zq::<Q>::one()); // w^{2d} = 1
        assert_eq!(omega.pow(D), -Zq::<Q>::one()); // w^d = -1
    }
}
