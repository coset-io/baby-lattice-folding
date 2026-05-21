use rand::rand_core::block;

use crate::ring::Ring;
use std::ops::{Add, Index, Mul, Range};

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
        Self { data, nrows, ncols }
    }

    pub fn from_flatten(nrows: usize, l: impl Into<Vec<R>>) -> Self {
        assert!(nrows != 0, "nrows must not be zero");
        let data: Vec<R> = l.into();
        let len_data = data.len();
        let ncols = len_data / nrows;
        assert!(
            ncols * nrows == len_data,
            "len_data is not divisible by nrows"
        );
        Self { data, nrows, ncols }
    }

    pub fn from_fn(nrows: usize, ncols: usize, mut f: impl FnMut(usize, usize) -> R) -> Self {
        let mut data = Vec::with_capacity(nrows * ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                data.push(f(i, j));
            }
        }
        Self { data, nrows, ncols }
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
    pub fn col(&self, j: usize) -> Vec<R> {
        assert!(j < self.ncols, "col index out of bounds");

        self.data
            .iter()
            .skip(j)
            .step_by(self.ncols)
            .cloned()
            .collect()
    }

    /// A.augment(B) = [A | B]
    pub fn augment(&self, other: &Self) -> Self {
        assert_eq!(self.nrows, other.nrows, "row mismatch");
        let new_ncols = self.ncols + other.ncols;
        let mut data = Vec::with_capacity(self.nrows * new_ncols);
        for i in 0..self.nrows {
            // Fill A into `data`
            for j in 0..self.ncols {
                data.push(self[(i, j)].clone());
            }
            // Fill B into `data`
            for j in 0..other.ncols {
                data.push(other[(i, j)].clone());
            }
        }

        Self {
            data,
            ncols: self.ncols + other.ncols,
            nrows: self.nrows,
        }
    }

    /// A.stack(B) = [A
    ///               B]
    pub fn stack(&self, other: &Self) -> Self {
        assert_eq!(self.ncols, other.ncols, "col mismatch");

        let mut data = self.data.clone();
        data.extend(other.data.clone());
        Self {
            data,
            nrows: self.nrows + other.nrows,
            ncols: self.ncols,
        }
    }

    pub fn transpose(&self) -> Self {
        Mat::from_fn(self.ncols, self.nrows, |i, j| {
            self.data[j * self.ncols + i].clone()
        })
    }

    /// Return a new Mat containing the rectangular region `rows × cols`.
    ///
    /// Both ranges are half-open (`start..end`). Panics if the end of either
    /// range exceeds the corresponding dimension.
    pub fn submatrix(&self, rows: Range<usize>, cols: Range<usize>) -> Self {
        assert!(rows.end <= self.nrows && cols.end <= self.ncols, "submatrix OOB");
        Mat::<R>::from_fn(
            rows.end - rows.start,
            cols.end - cols.start,
            // i \in [0, nrows), j \in [0, ncols)
            |i, j| self[(i + rows.start, j + cols.start)].clone()
        )
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

    /// Build a block-diagonal matrix from the supplied blocks. Off-diagonal
    /// entries are filled with `R::zero()`.
    ///
    /// Example: `block_diagonal(&[I_1, I_2, I_1])` is the 4×4 identity.
    pub fn block_diagonal(blocks: &[Mat<R>]) -> Self {
        let nrows_new: usize  = blocks.iter().map(|x| x.nrows).sum();
        let ncols_new: usize = blocks.iter().map(|x| x.ncols).sum();
        let mut data: Vec<R> = vec![R::zero();nrows_new * ncols_new];
        let mut cur_start_row: usize = 0;
        let mut cur_start_col: usize = 0;
        // Fill in each b in the diagonal of `data`.
        for b in blocks.iter() {
            for i in 0..b.nrows {
                for j in 0..b.ncols {
                    data[(cur_start_row + i) * ncols_new + cur_start_col + j] = b[(i, j)].clone();
                }
            }
            cur_start_row += b.nrows;
            cur_start_col += b.ncols;
        }
        Self {
            data,
            nrows: nrows_new,
            ncols: ncols_new,
        }
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
        let data = [z(1), z(2), z(3), z(4)];
        let nrows = 2;
        let m = Mat::<F>::from_flatten(nrows, &data);
        assert_eq!(&[m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]], &data);
    }

    #[test]
    #[should_panic(expected = "len_data is not divisible by nrows")]
    fn test_from_flatten_panics_nrows_not_dividing() {
        let data = [z(1), z(2), z(3), z(4)];
        let nrows = 3;
        let m = Mat::<F>::from_flatten(nrows, &data);
        assert_eq!(&[m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]], &data);
    }

    #[test]
    #[should_panic(expected = "nrows must not be zero")]
    fn test_from_flatten_panics_nrows_zero() {
        let data = [z(1), z(2), z(3), z(4)];
        let nrows = 0;
        let m = Mat::<F>::from_flatten(nrows, &data);
        assert_eq!(&[m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]], &data);
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
        assert_eq!(c[(0, 0)], r([11, 11, 11, 11]));
    }

    // ─── stack ───

    #[test]
    fn test_stack_basic() {
        // A: 2×3, B: 1×3 → 3×3
        let a = m(&[[1, 2, 3], [4, 5, 6]]);
        let b = m(&[[7, 8, 9]]);
        let c = a.stack(&b);
        assert_eq!(c.dimensions(), (3, 3));
        assert_eq!(c.row(0), &[z(1), z(2), z(3)]);
        assert_eq!(c.row(1), &[z(4), z(5), z(6)]);
        assert_eq!(c.row(2), &[z(7), z(8), z(9)]);
    }

    #[test]
    fn test_stack_preserves_ncols() {
        let a = m(&[[1, 2]]);
        let b = m(&[[3, 4], [5, 6]]);
        let c = a.stack(&b);
        assert_eq!(c.dimensions(), (3, 2));
    }

    #[test]
    #[should_panic(expected = "col mismatch")]
    fn test_stack_ncols_mismatch_panics() {
        let a = m(&[[1, 2, 3]]);
        let b = m(&[[4, 5]]);
        let _ = a.stack(&b);
    }

    #[test]
    fn test_stack_zero_rows_on_top() {
        // 0×3 stack 2×3 → 2×3 (the zero matrix vanishes on top)
        let empty = Mat::<F>::from_fn(0, 3, |_, _| z(0));
        let b = m(&[[1, 2, 3], [4, 5, 6]]);
        let c = empty.stack(&b);
        assert_eq!(c.dimensions(), (2, 3));
        assert_eq!(c.row(0), &[z(1), z(2), z(3)]);
        assert_eq!(c.row(1), &[z(4), z(5), z(6)]);
    }

    #[test]
    fn test_stack_zero_rows_on_bottom() {
        // 2×3 stack 0×3 → 2×3 (the zero matrix vanishes on bottom)
        let a = m(&[[1, 2, 3], [4, 5, 6]]);
        let empty = Mat::<F>::from_fn(0, 3, |_, _| z(0));
        let c = a.stack(&empty);
        assert_eq!(c.dimensions(), (2, 3));
        assert_eq!(c.row(0), &[z(1), z(2), z(3)]);
        assert_eq!(c.row(1), &[z(4), z(5), z(6)]);
    }

    #[test]
    fn test_stack_zero_rows_both() {
        // 0×3 stack 0×3 → 0×3
        let a = Mat::<F>::from_fn(0, 3, |_, _| z(0));
        let b = Mat::<F>::from_fn(0, 3, |_, _| z(0));
        let c = a.stack(&b);
        assert_eq!(c.dimensions(), (0, 3));
    }

    // ─── augment ───

    #[test]
    fn test_augment_basic() {
        // A = [1 2]     B = [5 6]    A | B = [1 2 5 6]
        //     [3 4]         [7 8]            [3 4 7 8]
        let a = m(&[[1, 2], [3, 4]]);
        let b = m(&[[5, 6], [7, 8]]);
        let c = a.augment(&b);
        assert_eq!(c.dimensions(), (2, 4));
        assert_eq!(c.row(0), &[z(1), z(2), z(5), z(6)]);
        assert_eq!(c.row(1), &[z(3), z(4), z(7), z(8)]);
    }

    #[test]
    fn test_augment_different_widths() {
        let a = m(&[[1], [2], [3]]); // 3 × 1
        let b = m(&[[4, 5], [6, 7], [8, 9]]); // 3 × 2
        let c = a.augment(&b);
        assert_eq!(c.dimensions(), (3, 3));
        assert_eq!(c.row(0), &[z(1), z(4), z(5)]);
        assert_eq!(c.row(1), &[z(2), z(6), z(7)]);
        assert_eq!(c.row(2), &[z(3), z(8), z(9)]);
    }

    #[test]
    #[should_panic(expected = "row mismatch")]
    fn test_augment_nrows_mismatch_panics() {
        let a = m(&[[1, 2], [3, 4]]);
        let b = m(&[[5, 6]]);
        let _ = a.augment(&b);
    }

    // ─── submatrix ───

    #[test]
    fn test_submatrix_basic() {
        // [1 2 3]
        // [4 5 6]
        // [7 8 9]   sub(1..3, 1..3) = [[5 6], [8 9]]
        let mat = m(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let sub = mat.submatrix(1..3, 1..3);
        assert_eq!(sub.dimensions(), (2, 2));
        assert_eq!(sub[(0, 0)], z(5));
        assert_eq!(sub[(0, 1)], z(6));
        assert_eq!(sub[(1, 0)], z(8));
        assert_eq!(sub[(1, 1)], z(9));
    }

    #[test]
    fn test_submatrix_full() {
        let mat = m(&[[1, 2], [3, 4]]);
        let sub = mat.submatrix(0..2, 0..2);
        assert_eq!(sub, mat);
    }

    #[test]
    fn test_submatrix_single_row() {
        let mat = m(&[[1, 2, 3], [4, 5, 6]]);
        let sub = mat.submatrix(1..2, 0..3);
        assert_eq!(sub.dimensions(), (1, 3));
        assert_eq!(sub.row(0), &[z(4), z(5), z(6)]);
    }

    #[test]
    fn test_submatrix_single_col() {
        let mat = m(&[[1, 2, 3], [4, 5, 6]]);
        let sub = mat.submatrix(0..2, 1..2);
        assert_eq!(sub.dimensions(), (2, 1));
        assert_eq!(sub[(0, 0)], z(2));
        assert_eq!(sub[(1, 0)], z(5));
    }

    #[test]
    #[should_panic(expected = "submatrix OOB")]
    fn test_submatrix_row_oob_panics() {
        let mat = m(&[[1, 2], [3, 4]]);
        let _ = mat.submatrix(0..3, 0..2); // row end > nrows
    }

    #[test]
    #[should_panic(expected = "submatrix OOB")]
    fn test_submatrix_col_oob_panics() {
        let mat = m(&[[1, 2], [3, 4]]);
        let _ = mat.submatrix(0..2, 0..3); // col end > ncols
    }

    // ─── block_diagonal ───

    #[test]
    fn test_block_diagonal_three_identities_equals_identity() {
        // diag(I_1, I_2, I_1) is the 4×4 identity (block_diag of identity blocks).
        let bd = Mat::<F>::block_diagonal(&[
            Mat::<F>::identity(1),
            Mat::<F>::identity(2),
            Mat::<F>::identity(1),
        ]);
        assert_eq!(bd, Mat::<F>::identity(4));
    }

    #[test]
    fn test_block_diagonal_mixed_dims() {
        // diag(2×3 block, 1×2 block) → 3×5 matrix
        let a = m(&[[1, 2, 3], [4, 5, 6]]); // 2×3
        let b = m(&[[7, 8]]);                // 1×2
        let bd = Mat::<F>::block_diagonal(&[a, b]);

        assert_eq!(bd.dimensions(), (3, 5));

        // top-left 2×3 = a
        assert_eq!(bd[(0, 0)], z(1));
        assert_eq!(bd[(0, 2)], z(3));
        assert_eq!(bd[(1, 0)], z(4));
        assert_eq!(bd[(1, 2)], z(6));

        // top-right 2×2 = zeros
        assert_eq!(bd[(0, 3)], z(0));
        assert_eq!(bd[(0, 4)], z(0));
        assert_eq!(bd[(1, 3)], z(0));
        assert_eq!(bd[(1, 4)], z(0));

        // bottom-left 1×3 = zeros
        assert_eq!(bd[(2, 0)], z(0));
        assert_eq!(bd[(2, 1)], z(0));
        assert_eq!(bd[(2, 2)], z(0));

        // bottom-right 1×2 = b
        assert_eq!(bd[(2, 3)], z(7));
        assert_eq!(bd[(2, 4)], z(8));
    }

    #[test]
    fn test_block_diagonal_single_block() {
        // diag(A) == A (degenerate case)
        let a = m(&[[1, 2], [3, 4]]);
        let bd = Mat::<F>::block_diagonal(&[a.clone()]);
        assert_eq!(bd, a);
    }
}
