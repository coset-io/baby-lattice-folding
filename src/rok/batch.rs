use crate::relations::LinRelation;

/// Batch evaluation statements into smaller statements.
///     E.g. \tilde f(r)      = s
///     \tilde f(\bar r) = \bar s
///     -> tilde f(r) + c \tilde f(\bar r) = s + c \bar s
pub fn rok_join<const Q: u64, const D: usize>(
    _lin: &LinRelation<Q, D>,
    _n_target_eval_rows: usize,
) -> LinRelation<Q, D> {
    todo!()
}
