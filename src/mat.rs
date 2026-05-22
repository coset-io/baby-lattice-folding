use crate::ring::Ring;
use std::ops::{Add, Index, Mul, Range};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mat<R> {
    data: Vec<R>,
    nrows: usize,
    ncols: usize,
}

impl<R> Mat<R> {
    /// Build a matrix from row vectors.
    ///
    /// Panics on empty outer vec — the column count is undeterminable
    /// from an empty input. For 0-row matrices, use `Mat::zero(0, ncols)`
    /// or `Mat::from_fn(0, ncols, _)` so `ncols` is explicit.
    /// `vec![vec![], vec![]]` (M rows of width 0) is fine — ncols = 0.
    pub fn new(rows: impl Into<Vec<Vec<R>>>) -> Self {
        let rows: Vec<Vec<R>> = rows.into();
        assert!(
            !rows.is_empty(),
            "Mat::new: empty rows — use Mat::zero(0, ncols) for 0×ncols",
        );
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

    pub fn iter(&self) -> impl Iterator<Item = &R> {
        self.data.iter()
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

    pub fn row(&self, i: usize) -> &[R] {
        assert!(i < self.nrows, "row index out of bounds");
        let start = i * self.ncols;
        &self.data[start..start + self.ncols]
    }
}

impl<R: Clone> Mat<R> {
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
        Self::from_fn(self.ncols, self.nrows, |i, j| {
            self.data[j * self.ncols + i].clone()
        })
    }

    /// Return a new Mat containing the rectangular region `rows × cols`.
    ///
    /// Both ranges are half-open (`start..end`). Panics if the end of either
    /// range exceeds the corresponding dimension.
    pub fn submatrix(&self, rows: Range<usize>, cols: Range<usize>) -> Self {
        assert!(
            rows.end <= self.nrows && cols.end <= self.ncols,
            "submatrix OOB"
        );
        Self::from_fn(
            rows.end - rows.start,
            cols.end - cols.start,
            // i \in [0, nrows), j \in [0, ncols)
            |i, j| self[(i + rows.start, j + cols.start)].clone(),
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
        let nrows_new: usize = blocks.iter().map(|x| x.nrows).sum();
        let ncols_new: usize = blocks.iter().map(|x| x.ncols).sum();
        let mut data: Vec<R> = vec![R::zero(); nrows_new * ncols_new];
        let mut cur_start_row: usize = 0;
        let mut cur_start_col: usize = 0;
        // Fill in each b in the diagonal of `data`.
        for b in blocks.iter() {
            for i in 0..b.nrows {
                for j in 0..b.ncols {
                    data[(cur_start_row + i) * ncols_new + cur_start_col + j] = b[(i, j)];
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

    /// Tensor product of two matrices. A: m_a x n_a, B: m_b x n_b
    /// A ⊗ B: (m_a * m_b) x (n_a * n_b)
    /// E.g. 
    /// A = [[1]   B = [[3, 4]]
    ///      [3]]
    /// A ⊗ B = [[3, 4],
    ///          [9, 12]]
    pub fn tensor_product(&self, other: &Self) -> Self {
        let new_nrows = self.nrows * other.nrows;
        let new_ncols = self.ncols * other.ncols;
        Self::from_fn(new_nrows, new_ncols, |i,j| {
            let i_in_self = i / other.nrows;
            let j_in_self = j / other.ncols;
            let i_in_other = i % other.nrows;
            let j_in_other = j % other.ncols;
            // self[(i_in_self, j_in_self)] * other[(i_in_other, j_in_other)]
            self.data[i_in_self * self.ncols + j_in_self] * other.data[i_in_other * other.ncols + j_in_other]
        })
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

    /// Matrix multiplication
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

impl<R: std::ops::Mul<Output=R> + Clone> Mul<R> for Mat<R> {
    type Output = Self;

    /// Scalar multiplication
    fn mul(self, rhs: R) -> Self {
        Mat::from_fn(self.nrows, self.ncols, |i, j| {
            // TODO: a lot of clone() ...
            self.data[i * self.ncols + j].clone() * rhs.clone()
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
    #[should_panic(expected = "empty rows")]
    fn test_new_empty_panics() {
        // Empty outer Vec is ambiguous (ncols undeterminable) — must panic.
        let _: Mat<F> = Mat::new(Vec::<Vec<F>>::new());
    }

    #[test]
    fn test_new_zero_cols_from_empty_inner_rows() {
        // M × 0: outer has rows, inner rows have width 0 → ncols = 0 is determined.
        let mat: Mat<F> = Mat::new(vec![vec![], vec![], vec![]]);
        assert_eq!(mat.dimensions(), (3, 0));
    }

    #[test]
    fn test_from_fn_matches_new() {
        let by_new = m(&[[0, 1, 2], [10, 11, 12]]);
        let by_fn = Mat::<F>::from_fn(2, 3, |i, j| z((i * 10 + j) as u64));
        assert_eq!(by_new, by_fn);
    }

    // ─── 0-dim matrices ───

    #[test]
    fn test_from_fn_zero_rows() {
        // 0 × m: closure never invoked, but ncols is preserved.
        let mat = Mat::<F>::from_fn(0, 5, |_, _| panic!("must not be called"));
        assert_eq!(mat.dimensions(), (0, 5));
    }

    #[test]
    fn test_from_fn_zero_cols() {
        let mat = Mat::<F>::from_fn(3, 0, |_, _| panic!("must not be called"));
        assert_eq!(mat.dimensions(), (3, 0));
    }

    #[test]
    fn test_zero_with_zero_rows() {
        let mat = Mat::<F>::zero(0, 4);
        assert_eq!(mat.dimensions(), (0, 4));
    }

    #[test]
    fn test_zero_with_zero_cols() {
        let mat = Mat::<F>::zero(2, 0);
        assert_eq!(mat.dimensions(), (2, 0));
    }

    #[test]
    fn test_stack_onto_zero_row_keeps_ncols() {
        // 0 × 3 stacked on 2 × 3 = 2 × 3 (caller can drop the empty top half).
        let top = Mat::<F>::zero(0, 3);
        let bot = m(&[[1, 2, 3], [4, 5, 6]]);
        let stacked = top.stack(&bot);
        assert_eq!(stacked.dimensions(), (2, 3));
        assert_eq!(stacked, bot);
    }

    #[test]
    #[should_panic(expected = "col mismatch")]
    fn test_stack_zero_x_zero_onto_real_matrix_panics() {
        // 0 × 0 ≠ "any ncols" — stack must reject this.
        let zero_zero = Mat::<F>::from_fn(0, 0, |_, _| unreachable!());
        let real = m(&[[1, 2, 3]]);
        let _ = zero_zero.stack(&real);
    }

    #[test]
    fn test_augment_with_zero_col_keeps_nrows() {
        // (2 × 0) augment (2 × 3) = 2 × 3.
        let left = Mat::<F>::zero(2, 0);
        let right = m(&[[1, 2, 3], [4, 5, 6]]);
        let aug = left.augment(&right);
        assert_eq!(aug.dimensions(), (2, 3));
        assert_eq!(aug, right);
    }

    #[test]
    fn test_transpose_zero_row() {
        // (0 × 5)^T = (5 × 0).
        let mat = Mat::<F>::zero(0, 5);
        let t = mat.transpose();
        assert_eq!(t.dimensions(), (5, 0));
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

    // ─── Scalar Mul (Mat * R) ───

    #[test]
    fn test_scalar_mul_2x2() {
        // [1 2] * 3 = [3  6]
        // [3 4]       [9 12]
        let a = m(&[[1, 2], [3, 4]]);
        let expected = m(&[[3, 6], [9, 12]]);
        assert_eq!(a * z(3), expected);
    }

    #[test]
    fn test_scalar_mul_with_mod_reduction() {
        // Every entry × 4, mod 17: 5·4=20→3, 9·4=36→2, 13·4=52→1
        let a = m(&[[5, 9], [13, 0]]);
        let expected = m(&[[3, 2], [1, 0]]);
        assert_eq!(a * z(4), expected);
    }

    #[test]
    fn test_scalar_mul_by_zero_yields_zero_matrix() {
        let a = m(&[[3, 5], [7, 11]]);
        let (nrows, ncols) = a.dimensions();
        assert_eq!(a * z(0), Mat::<F>::zero(nrows, ncols));
    }

    #[test]
    fn test_scalar_mul_by_one_is_identity() {
        // c=1 leaves the matrix unchanged (acts as ring identity entry-wise).
        let a = m(&[[3, 5, 7], [11, 13, 2]]);
        assert_eq!(a.clone() * z(1), a);
    }

    #[test]
    fn test_scalar_mul_rectangular_shape_preserved() {
        // Scalar mul never changes dimensions — only entry values.
        let a = m(&[[1, 2, 3], [4, 5, 6]]);
        let result = a.clone() * z(2);
        assert_eq!(result.dimensions(), a.dimensions());
    }

    #[test]
    fn test_scalar_mul_distributes_over_add() {
        // c · (A + B) == c·A + c·B
        let a = m(&[[1, 2], [3, 4]]);
        let b = m(&[[5, 6], [7, 8]]);
        let c = z(3);
        let lhs = (a.clone() + b.clone()) * c;
        let rhs = (a * c) + (b * c);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_scalar_mul_compatible_with_matrix_mul() {
        // (c · A) · B == c · (A · B) — scalar slides through matrix mul.
        let a = m(&[[1, 2], [3, 4]]);
        let b = m(&[[5, 6], [7, 8]]);
        let c = z(3);
        let lhs = (a.clone() * c) * b.clone();
        let rhs = (a * b) * c;
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_scalar_mul_over_rq() {
        // Mat<Rq<Q,D>> * Rq<Q,D>: every poly entry multiplied by an Rq scalar.
        // (1 + X) entries scaled by 2 → (2 + 2X) entries.
        let r = |c: [u64; D]| Ring4::new(c.map(z));
        let a = Mat::new(vec![
            vec![r([1, 1, 0, 0]), r([1, 1, 0, 0])],
            vec![r([1, 1, 0, 0]), r([1, 1, 0, 0])],
        ]);
        let scalar = r([2, 0, 0, 0]); // constant poly 2
        let result = a * scalar;
        let expected_entry = r([2, 2, 0, 0]); // 2 · (1 + X) = 2 + 2X
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(result[(i, j)], expected_entry, "mismatch at ({i}, {j})");
            }
        }
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
        let b = m(&[[7, 8]]); // 1×2
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

    // ─── tensor_product (Kronecker product) ───

    /// Docstring example: A = [[1],[3]], B = [[3,4]] → A⊗B = [[3,4],[9,12]].
    #[test]
    fn test_tensor_product_docstring_example() {
        let a = m(&[[1], [3]]);
        let b = m(&[[3, 4]]);
        let result = a.tensor_product(&b);
        assert_eq!(result, m(&[[3, 4], [9, 12]]));
    }

    /// Dim arithmetic: (A.rows × A.cols) ⊗ (B.rows × B.cols)
    ///                  = (A.rows·B.rows) × (A.cols·B.cols)
    #[test]
    fn test_tensor_product_dimensions() {
        let a = m(&[[1, 2, 3], [4, 5, 6]]); // 2×3
        let b = m(&[[1, 2], [3, 4]]);       // 2×2
        let result = a.tensor_product(&b);   // expect 4×6
        assert_eq!(result.dimensions(), (4, 6));
    }

    /// 1×1 identity is the tensor identity on either side.
    #[test]
    fn test_tensor_product_identity_one_is_neutral() {
        let a = m(&[[1, 2, 3], [4, 5, 6]]);
        let i1 = Mat::<F>::identity(1);
        assert_eq!(i1.tensor_product(&a), a, "I_1 ⊗ A = A");
        assert_eq!(a.tensor_product(&i1), a, "A ⊗ I_1 = A");
    }

    /// I_n ⊗ A == block_diag(A, A, ..., A) with n copies — exactly the structure
    /// `rok_rp` builds for Ĵ = I_{m/m_rp} ⊗ J. Cross-checks tensor_product against
    /// block_diagonal (two independent impls).
    #[test]
    fn test_tensor_product_identity_left_equals_block_diagonal() {
        let a = m(&[[1, 2], [3, 4]]);
        let n = 3;
        let i_n = Mat::<F>::identity(n);
        let lhs = i_n.tensor_product(&a);
        let rhs = Mat::<F>::block_diagonal(&[a.clone(), a.clone(), a]);
        assert_eq!(lhs, rhs);
    }

    /// Associativity: (A ⊗ B) ⊗ C == A ⊗ (B ⊗ C).
    #[test]
    fn test_tensor_product_associativity() {
        let a = m(&[[1, 2]]);       // 1×2
        let b = m(&[[3], [4]]);     // 2×1
        let c = m(&[[5, 6]]);       // 1×2
        let lhs = a.tensor_product(&b).tensor_product(&c);
        let rhs = a.tensor_product(&b.tensor_product(&c));
        assert_eq!(lhs, rhs);
    }

    /// Mixed-product identity: (A ⊗ B)(C ⊗ D) == (A·C) ⊗ (B·D).
    /// This is the algebraic property that makes Kronecker products useful —
    /// catches any subtle index bug in the impl.
    #[test]
    fn test_tensor_product_mixed_product() {
        let a = m(&[[1, 2], [3, 4]]);
        let b = m(&[[5, 6], [7, 8]]);
        let c = m(&[[1, 0], [0, 1]]);
        let d = m(&[[2, 0], [0, 3]]);
        // dim: A.cols == C.rows, B.cols == D.rows  ✓
        let lhs = a.clone().tensor_product(&b.clone()) * c.clone().tensor_product(&d.clone());
        let rhs = (a * c).tensor_product(&(b * d));
        assert_eq!(lhs, rhs);
    }
}
