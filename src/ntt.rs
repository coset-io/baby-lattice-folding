use super::{zq::Zq, poly::Poly};

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


/// R_q = Z_q[X]/(X^d+1). X^d + 1 = 0 -> X^d = -1 \mod q
/// -> X^{2d} = 1. Assume q is prime, Z_q^* is a cyclic group with order q-1
/// i.e. \forall g \in Z_q^*, g^{q-1} = 1. Since g^{(q-1)/(2d)}^{2d} = 1,
/// for g to exist 2d must divide (q-1).
pub fn find_primitive_2d_root_of_unity<const Q: u64>(d: u64) -> Zq<Q> {

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
        let is_generator = factors.iter().all(|&p| g.pow(order / p) != Zq::<Q>::one());
        if is_generator {
            // if g is a generator -> o(g) = q-1 -> o(g^{(q-1)/2d}) = 2d
            return g.pow(order / (2 * d));
        }
    }
    panic!("no multiplicative generator found for Q={Q}")
}


/// NTT: here we assume coeffs can be split *completely* for simplicity and efficiency
/// in the split fields.
/// This is implemented according to this great article https://electricdusk.com/ntt.html
/// This requires {d} to be a power of two.
pub fn ntt<const Q: u64>(coeffs: Vec<Zq<Q>>, psi: Zq<Q>, psi_power: u64) -> Vec<Zq<Q>> {
    let d = coeffs.len();
    assert!(d.is_power_of_two(), "d should be power of two to split completely: d={d}");
    assert!((Q-1).is_multiple_of(2*d as u64));

    // Terminal condition: when d = 1, it's the last split. Just returns
    // the constant term.
    if d == 1 {
        return vec![coeffs[0]];
    }

    // E.g. d=256, root here is \psi^{128} since X^{256}+1 = (X^{128} - 1)(X^{128} + 1)
    let root= psi.pow(psi_power);
    // Here is the "butterfly" part
    // E.g. we're at a \in Z_q[X] / (X^256+1) and we're gonna split to
    // a_l \in Z_q[X]/(X^128 - \psi^128), a_r \in Z_q[X]/(X^128 + \psi^128).
    // We just let replace all X^128=\psi^128  in a to become a_l,
    //                         X^128=-\psi^128 in a to become a_r.
    // Then,
    //      a_l[0] = a[0] + psi^{128} * a[128]
    //      a_r[0] = a[0] - psi^{128} * a[128]
    // Since `a[0]` and `psi^{128} * a[128]` are reused for a_l and a_r, just different
    // operator before the latter term.
    // We can draw it as a butterfly.

    let mut a_l: Vec<Zq<Q>> = Vec::new();
    let mut a_r: Vec<Zq<Q>> = Vec::new();

    for i in 0..(d/2) {
        a_l.push(coeffs[i] + root * coeffs[i + d/2]);
        a_r.push(coeffs[i] - root * coeffs[i + d/2]);
    }

    // Split the left/right poly all the way down and get the results.
    let a_l_coeffs = ntt(a_l, psi, psi_power / 2);
    let a_r_coeffs = ntt(a_r, psi, psi_power / 2 + (d/2) as u64);
    a_l_coeffs.into_iter().chain(a_r_coeffs).collect()
}

pub fn intt<const Q: u64>(evals: Vec<Zq<Q>>) -> Vec<Zq<Q>> {
    todo!()
}

#[cfg(test)]
mod tests {

    use super::*;

    const Q: u64 = 17;
    const D: u64 = 4;
    type F = Zq<Q>;

    fn setup() -> Zq<Q> {
        let psi = find_primitive_2d_root_of_unity::<Q>(D);
        println!("psi={:?}", psi);
        psi
    }

    #[test]
    fn test_primitive_2d_root_of_unity() {
        let psi = setup();
        assert_eq!(psi.pow(2 * D), F::one()); // w^{2d} = 1
        assert_eq!(psi.pow(D), -F::one()); // w^d = -1
    }

    // Sage test vectors: q=17, d=4, negacyclic NTT (X^d+1)
    // coeffs [16, 3, 0, 14] <-> evals [15, 0, 0, 15]
    #[test]
    fn test_ntt_forward() {
        let psi = setup();
        let d = 4;
        let coeffs = vec![F::new(16), F::new(3), F::new(0), F::new(14)];
        let expected_evals = vec![F::new(15), F::new(0), F::new(0), F::new(15)];

        let odd_powers: Vec<_> = (0..d).map(|k| psi.pow(2*k as u64 + 1)).collect();
        println!("roots: {:?}", odd_powers);

        let a = Poly::new(coeffs.clone());
        let evals: Vec<_> = odd_powers.iter().map(|w| a.clone().eval(w.value())).collect();
        println!("evals: {:?}", evals);
        assert_eq!(ntt::<Q>(coeffs, psi, d/2), expected_evals);
    }


    #[test]
    fn test_intt_backward() {
        let evals = vec![F::new(15), F::new(0), F::new(0), F::new(15)];
        let expected_coeffs = vec![F::new(16), F::new(3), F::new(0), F::new(14)];
        assert_eq!(intt::<Q>(evals), expected_coeffs);
    }

    #[test]
    fn test_ntt_intt_roundtrip() {
        type F = Zq<Q>;
        let psi = setup();
        let d = 4;
        let coeffs = vec![F::new(16), F::new(3), F::new(0), F::new(14)];
        // assert_eq!(intt::<Q>(ntt::<Q>(coeffs, psi, d)), coeffs);
    }
}
