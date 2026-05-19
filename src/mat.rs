use crate::ring::Ring;
use std::ops::{Add, Index, Mul};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mat<R> {
    data: Vec<R>,
    nrows: usize,
    ncols: usize,
}

impl<R> Mat<R> {
    pub fn new(rows: impl Into<Vec<Vec<R>>>) -> Self {
        let rows: Vec<Vec<R>> = rows.into();
        assert!(!rows.is_empty(), "Mat::new requires at least one row");
        let nrows = rows.len();
        let ncols = rows[0].len();
        assert!(
            rows.iter().all(|r| r.len() == ncols),
            "Mat::new requires all rows to have the same length",
        );
        let data = rows.into_iter().flatten().collect();
        Self {
            data,
            nrows,
            ncols,
        }
    }

    pub fn from_flatten(nrows: usize, l: impl Into<Vec<R>>) -> Self {
        assert!(nrows != 0, "nrows must not be zero");
        let data: Vec<R> = l.into();
        let len_data = data.len();
        let ncols = len_data / nrows;
        assert!(ncols * nrows == len_data, "len_data is not divisible by nrows");
        Self {
            data,
            nrows,
            ncols,
        }
    }

    pub fn from_fn(nrows: usize, ncols: usize, mut f: impl FnMut(usize, usize) -> R) -> Self {
        let mut data = Vec::with_capacity(nrows * ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                data.push(f(i, j));
            }
        }
        Self {
            data,
            nrows,
            ncols,
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

impl<R: Clone> Mat<R> {
    pub fn row(&self, i: usize) -> &[R] {
        assert!(i < self.nrows, "row index out of bounds");
        let start = i * self.ncols;
        &self.data[start..start + self.ncols]
    }

    /// Returns column `j` as an owned `Vec<R>`.
    ///
    /// Row-major storage means column `j`'s elements live at offsets
    /// `j, j + ncols, j + 2*ncols, ...` — they are NOT contiguous, so
    /// we cannot return a `&[R]` slice (a slice must be contiguous in
    /// memory). That is why this method clones into an owned `Vec`
    /// and why the bound `R: Clone` is required.
    pub fn col(&self, j: usize) -> Vec<R> {
        assert!(j < self.ncols, "col index out of bounds");

        self.data.iter().skip(j).step_by(self.ncols).cloned().collect()
    }

    pub fn transpose(&self) -> Self {
        Mat::from_fn(self.ncols, self.nrows, |i, j| {
            self.data[j * self.ncols + i].clone()
        })
    }
}

impl<R> Index<(usize, usize)> for Mat<R> {
    type Output = R;

    fn index(&self, (i, j): (usize, usize)) -> &R {
        assert!(i < self.nrows && j < self.ncols, "index out of bounds");
        &self.data[i * self.ncols + j]
    }
}

impl<R: Ring> Mat<R> {
    pub fn zero(nrows: usize, ncols: usize) -> Self {
        Mat::from_fn(nrows, ncols, |_, _| R::zero())
    }

    pub fn identity(n: usize) -> Self {
        Mat::from_fn(n, n, |i, j| if i == j { R::one() } else { R::zero() })
    }
}

impl<R: Ring> Add for Mat<R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        assert_eq!(
            self.dimensions(),
            rhs.dimensions(),
            "Mat add: dimension mismatch",
        );
        let data = self
            .data
            .into_iter()
            .zip(rhs.data)
            .map(|(a, b)| a + b)
            .collect();
        Mat {
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<R: Ring> Mul for Mat<R> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        assert_eq!(
            self.ncols, rhs.nrows,
            "Mat mul: lhs.ncols ({}) must equal rhs.nrows ({})",
            self.ncols, rhs.nrows,
        );
        let n = self.nrows;
        let m = rhs.ncols;
        let k = self.ncols;
        Mat::from_fn(n, m, |i, j| {
            let mut sum = R::zero();
            for l in 0..k {
                sum = sum + self.data[i * k + l] * rhs.data[l * m + j];
            }
            sum
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::{Rq, RqNtt};
    use crate::zq::Zq;

    const Q: u64 = 17;
    const D: usize = 4;
    type F = Zq<Q>;
    type Ring4 = Rq<Q, D>;
    type Ring4Ntt = RqNtt<Q, D>;

    fn z(v: u64) -> F {
        F::new(v)
    }

    fn m<const C: usize>(rows: &[[u64; C]]) -> Mat<F> {
        let v: Vec<Vec<F>> = rows
            .iter()
            .map(|r| r.iter().map(|&v| z(v)).collect())
            .collect();
        Mat::new(v)
    }

    // ─── new / from_fn / shape ───

    #[test]
    fn test_new_basic() {
        let mat = m(&[[1, 2, 3], [4, 5, 6]]);
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 3);
        assert_eq!(mat.dimensions(), (2, 3));
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_new_mismatched_row_lengths_panics() {
        let _ = Mat::new(vec![vec![z(1), z(2)], vec![z(3)]]);
    }

    #[test]
    #[should_panic(expected = "at least one row")]
    fn test_new_empty_panics() {
        let _: Mat<F> = Mat::new(Vec::<Vec<F>>::new());
    }

    #[test]
    fn test_from_fn_matches_new() {
        let by_new = m(&[[0, 1, 2], [10, 11, 12]]);
        let by_fn = Mat::<F>::from_fn(2, 3, |i, j| z((i * 10 + j) as u64));
        assert_eq!(by_new, by_fn);
    }

    #[test]
    fn test_from_flatten() {
        let data = [z(1), z(2) ,z(3), z(4)];
        let nrows = 2;
        let m = Mat::<F>::from_flatten(nrows, &data);
        assert_eq!(&[m[(0,0)], m[(0,1)], m[(1,0)], m[(1,1)]], &data);
    }

    #[test]
    #[should_panic(expected = "len_data is not divisible by nrows")]
    fn test_from_flatten_panics_nrows_not_dividing() {
        let data = [z(1), z(2) ,z(3), z(4)];
        let nrows = 3;
        let m = Mat::<F>::from_flatten(nrows, &data);
        assert_eq!(&[m[(0,0)], m[(0,1)], m[(1,0)], m[(1,1)]], &data);
    }

    #[test]
    #[should_panic(expected = "nrows must not be zero")]
    fn test_from_flatten_panics_nrows_zero() {
        let data = [z(1), z(2) ,z(3), z(4)];
        let nrows = 0;
        let m = Mat::<F>::from_flatten(nrows, &data);
        assert_eq!(&[m[(0,0)], m[(0,1)], m[(1,0)], m[(1,1)]], &data);
    }

    // ─── Index / row / col ───

    #[test]
    fn test_index() {
        let mat = m(&[[1, 2, 3], [4, 5, 6]]);
        assert_eq!(mat[(0, 0)], z(1));
        assert_eq!(mat[(0, 2)], z(3));
        assert_eq!(mat[(1, 1)], z(5));
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_oob_panics() {
        let mat = m(&[[1, 2], [3, 4]]);
        let _ = mat[(2, 0)];
    }

    #[test]
    fn test_row() {
        let mat = m(&[[1, 2, 3], [4, 5, 6]]);
        assert_eq!(mat.row(0), &[z(1), z(2), z(3)]);
        assert_eq!(mat.row(1), &[z(4), z(5), z(6)]);
    }

    #[test]
    fn test_col_basic() {
        let mat = m(&[[1, 2, 3], [4, 5, 6]]);
        assert_eq!(mat.col(0), vec![z(1), z(4)]);
        assert_eq!(mat.col(1), vec![z(2), z(5)]);
        assert_eq!(mat.col(2), vec![z(3), z(6)]);
    }

    #[test]
    fn test_col_single_column() {
        let mat = m(&[[7], [8], [9]]);
        assert_eq!(mat.col(0), vec![z(7), z(8), z(9)]);
    }

    // ─── transpose ───

    #[test]
    fn test_transpose_shape() {
        let mat = m(&[[1, 2, 3], [4, 5, 6]]);
        let t = mat.transpose();
        assert_eq!(t.dimensions(), (3, 2));
    }

    #[test]
    fn test_transpose_values() {
        let mat = m(&[[1, 2, 3], [4, 5, 6]]);
        let t = mat.transpose();
        assert_eq!(t.row(0), &[z(1), z(4)]);
        assert_eq!(t.row(1), &[z(2), z(5)]);
        assert_eq!(t.row(2), &[z(3), z(6)]);
    }

    #[test]
    fn test_transpose_involution() {
        let mat = m(&[[1, 2, 3], [4, 5, 6]]);
        assert_eq!(mat.transpose().transpose(), mat);
    }

    // ─── zero / identity ───

    #[test]
    fn test_zero() {
        let mat = Mat::<F>::zero(2, 3);
        assert_eq!(mat.dimensions(), (2, 3));
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(mat[(i, j)], F::zero());
            }
        }
    }

    #[test]
    fn test_identity() {
        let id = Mat::<F>::identity(3);
        assert_eq!(id.dimensions(), (3, 3));
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { F::one() } else { F::zero() };
                assert_eq!(id[(i, j)], expected);
            }
        }
    }

    // ─── Add ───

    #[test]
    fn test_add() {
        let a = m(&[[1, 2], [3, 4]]);
        let b = m(&[[5, 6], [7, 8]]);
        let c = m(&[[6, 8], [10, 12]]);
        assert_eq!(a + b, c);
    }

    #[test]
    fn test_add_with_mod() {
        // (10 + 8) mod 17 = 1, (16 + 2) mod 17 = 1, etc.
        let a = m(&[[10, 16], [0, 5]]);
        let b = m(&[[8, 2], [1, 13]]);
        let c = m(&[[1, 1], [1, 1]]);
        assert_eq!(a + b, c);
    }

    #[test]
    fn test_add_zero_is_identity() {
        let a = m(&[[3, 5], [7, 11]]);
        let zm = Mat::<F>::zero(2, 2);
        assert_eq!(a.clone() + zm, a);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_add_dim_mismatch_panics() {
        let a = m(&[[1, 2], [3, 4]]);
        let b = m(&[[1, 2, 3]]);
        let _ = a + b;
    }

    // ─── Mul ───

    #[test]
    fn test_mul_2x2() {
        // [1 2]   [5 6]   [19 22]
        // [3 4] * [7 8] = [43 50]
        // mod 17: 19→2, 22→5, 43→9, 50→16
        let a = m(&[[1, 2], [3, 4]]);
        let b = m(&[[5, 6], [7, 8]]);
        let c = m(&[[2, 5], [9, 16]]);
        assert_eq!(a * b, c);
    }

    #[test]
    fn test_mul_rectangular() {
        // (2x3) * (3x1)
        // [1 2 3]   [1]   [14]   mod 17 → [14]
        // [4 5 6] * [2] = [32]            [15]
        //           [3]
        let a = m(&[[1, 2, 3], [4, 5, 6]]);
        let b = m(&[[1], [2], [3]]);
        let c = m(&[[14], [15]]);
        let result = a * b;
        assert_eq!(result.dimensions(), (2, 1));
        assert_eq!(result, c);
    }

    #[test]
    fn test_mul_identity() {
        let a = m(&[[3, 5, 7], [11, 13, 2]]);
        let id_left = Mat::<F>::identity(2);
        let id_right = Mat::<F>::identity(3);
        assert_eq!(id_left * a.clone(), a);
        assert_eq!(a.clone() * id_right, a);
    }

    #[test]
    fn test_mul_associativity() {
        let a = m(&[[1, 2], [3, 4]]);
        let b = m(&[[5, 6], [7, 8]]);
        let c = m(&[[9, 10], [11, 12]]);
        let ab_c = (a.clone() * b.clone()) * c.clone();
        let a_bc = a * (b * c);
        assert_eq!(ab_c, a_bc);
    }

    #[test]
    #[should_panic(expected = "must equal rhs.nrows")]
    fn test_mul_dim_mismatch_panics() {
        let a = m(&[[1, 2], [3, 4]]);
        let b = m(&[[5, 6, 7], [8, 9, 10], [11, 12, 13]]);
        // a is 2x2, b is 3x3 → 2 != 3, should panic.
        let _ = a * b;
    }

    // ─── Works over Rq too ───

    #[test]
    fn test_works_over_rq() {
        // Verify Mat<Rq<Q,D>> compiles and basic ops behave.
        let r = |c: [u64; D]| Ring4::new(c.map(z));
        let a = Mat::new(vec![vec![r([1, 0, 0, 0]), r([0, 1, 0, 0])]]);
        let b = Mat::new(vec![vec![r([0, 1, 0, 0])], vec![r([1, 0, 0, 0])]]);
        let c = a * b;
        // (1) * (X) + (X) * (1) = 2X
        assert_eq!(c.dimensions(), (1, 1));
        assert_eq!(c[(0, 0)], r([0, 2, 0, 0]));
    }


    #[test]
    fn test_works_over_rq_ntt() {
        // Verify Mat<Rq<Q,D>> compiles and basic ops behave.
        let r = |c: [u64; D]| Ring4Ntt::new(c.map(z));
        // a = [[1, 2]]
        let a = Mat::new(vec![vec![r([1, 1, 1, 1]), r([2, 2, 2, 2])]]);
        // b = [[3], [4]]
        let b = Mat::new(vec![vec![r([3, 3, 3, 3])], vec![r([4, 4, 4, 4])]]);
        let c = a * b;
        // (1) * (X) + (X) * (1) = 2X
        assert_eq!(c.dimensions(), (1, 1));
        // 1*3+2*4 = 11
        assert_eq!(c[(0,0)], r([11, 11, 11, 11]));
    }

}
