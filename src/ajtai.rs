use rand::Rng;

use super::ring::RqNtt;

// Generate a random matrix (kappa x m) of Rq, in evaluation form.
pub fn setup<const Q: u64, const D: usize>(
    kappa: usize,
    m: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<RqNtt<Q, D>>> {
    let mut mat: Vec<Vec<RqNtt<Q, D>>> = Vec::new();
    for _ in 0..kappa {
        let mut row: Vec<RqNtt<Q, D>> = Vec::new();
        for _ in 0..m {
            row.push(RqNtt::random(rng));
        }
        mat.push(row);
    }
    mat
}


pub fn commit<const Q: u64, const D: usize>(
    a: &[Vec<RqNtt<Q, D>>],
    z: &[RqNtt<Q, D>],
) -> Vec<RqNtt<Q, D>> {
    let kappa = a.len();
    assert!(kappa > 0);

    let m = a[0].len();
    assert!(z.len() == m);

    let mut res: Vec<RqNtt<Q, D>> = Vec::new();
    for row in a.iter() {
        let mut s = RqNtt::<Q, D>::zero();
        for j in 0..m {
            s = s + row[j] * z[j];
        }
        res.push(s);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: u64 = 17;
    const D: usize = 4;

    const K: usize = 2;
    const M: usize = 4;

    // testing

    fn get_rng() -> impl Rng {
        // StdRng::seed_from_u64(42)
        // real rng
        rand::rng()
    }

    fn gen_rand_matrix() -> Vec<Vec<RqNtt<Q, D>>> {
        let mut rng = get_rng();
        setup::<Q, D>(K, M, &mut rng)
    }

    fn gen_rand_vector() -> Vec<RqNtt<Q, D>> {
        let mut rng = get_rng();
        let mut v: Vec<RqNtt<Q, D>> = Vec::new();
        for _ in 0..M {
            v.push(RqNtt::random(&mut rng));
        }
        v
    }

    #[test]
    fn test_setup() {
        let a = gen_rand_matrix();
        println!("{:?}", a);
        // todo!()
    }

    #[test]
    fn test_commit() {
        let a = gen_rand_matrix();
        let z1 = gen_rand_vector();
        let c1 = commit(&a, &z1);

        let z2 = gen_rand_vector();
        let c2 = commit(&a, &z2);

        // Addition homomorphism:
        //  - c1 = A z1, c2 = A z2, c1+c2 = A (z1 + z2)
        let z_sum: Vec<_> = z1.iter().zip(z2.iter()).map(|(a, b)| *a + *b).collect();

        let c_sum: Vec<_> = c1.iter().zip(c2.iter()).map(|(a, b)| *a + *b).collect();
        assert_eq!(commit(&a, &z_sum), c_sum);
    }
}
