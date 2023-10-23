use pest::{iterators::Pair, Parser};
use pest_derive::Parser;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

#[derive(Parser)]
#[grammar = "n.pest"]
pub struct N;

/// parses a tensor from a string; used by Tensor::from() and tensor!() internally
fn parse_tensor<T: Field>(s: &str) -> Tensor<T> {
    let mut n = N::parse(Rule::tensor, s).unwrap();
    let mut shape: Vec<usize> = Vec::new();
    let mut value: Vec<T> = Vec::new();

    fn parse_value<U: Field>(
        pair: Pair<Rule>,
        depth: isize,
        shape: &mut Vec<usize>,
        value: &mut Vec<U>,
    ) {
        match pair.as_rule() {
            Rule::tensor => {
                if depth > 0 {
                    if shape.get((depth as usize) - 1) == None {
                        shape.push(1);
                    } else {
                        shape[(depth as usize) - 1] += 1;
                    }
                }

                for p in pair.into_inner() {
                    parse_value(p, depth + 1, shape, value);
                }
            }

            Rule::scalar => unsafe {
                value.push(pair.as_str().parse().unwrap_unchecked());
            },

            _ => {}
        }
    }
    parse_value(n.next().unwrap(), 0, &mut shape, &mut value);
    let mut dimensions;

    if shape.len() != 0 {
        dimensions = value.len() / shape.last().unwrap();
        for i in 0..shape.len() {
            for j in (0..i).rev() {
                shape[i] = shape[i] / shape[j];
            }
        }
    } else {
        dimensions = value.len();
    }

    shape.push(dimensions);

    let mut rank = shape.len();

    if dimensions == 1 || (rank == 2 && shape[0] == 1) {
        rank -= 1;
    }

    let shape_len = shape.len();

    if shape.len() > 1 {
        if shape[shape_len - 1] == 1 || shape[shape_len - 2] == 1 {
            let row = shape[shape_len - 1];
            let col = shape[shape_len - 2];
            dimensions = std::cmp::max(row, col);
        }
    }

    Tensor {
        rank,
        shape,
        dimensions,
        value,
    }
}

/// a trait that encapsulates the traits needed for the generic-type
/// used for tensors; integer, floats (i32, i64, f64, u32, etc)
pub trait Field: Add + Mul + Sub + Div + FromStr + Debug + Copy {}
impl<T> Field for T
where
    T: Add + Mul + Sub + Div + FromStr + Debug + Copy,
    <T as FromStr>::Err: Debug,
{
}

#[derive(Debug)]
pub struct Tensor<T: Field> {
    pub rank: usize,
    pub shape: Vec<usize>,
    pub dimensions: usize,
    pub value: Vec<T>,
}

impl<T: Field> Tensor<T> {
    pub fn shape(&self) -> &Vec<usize> {
        &(self.shape)
    }
}

impl<T: Field> FromStr for Tensor<T> {
    // FIXME: error handling
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s
            .replace(" ", "")
            .replace("\n", "")
            .replace("\t", "")
            .replace("\r", "");
        Ok(parse_tensor(&*s))
    }

    type Err = String;
}

impl<T: Field> From<&str> for Tensor<T> {
    fn from(value: &str) -> Self {
        Self::from_str(value).unwrap()
    }
}

impl<T: Field> Tensor<T> {
    pub fn flat_index(&self, index: usize) -> T {
        self.value[index]
    }
}

#[macro_export]
macro_rules! tensor {
    ($v:expr) => {
        Tensor::<f64>::from(stringify!($v))
    };

    ($t:ty, $v:expr) => {
        Tensor::<$t>::from(stringify!($v))
    };
}

#[macro_export]
macro_rules! index {
    ($t:ident, $($i:tt)*) => {
        {
            let indices = [$( $i )*];
            let shape = $t.shape();
            let mut index = 0;

            if shape.len() - indices.len() != 0 {
                panic!("bad indices");
            }

            for i in 0..indices.len() {
                let mut tmp = indices[i];
                for j in i..shape.len() {
                    if i != j {
                        tmp += tmp * (shape[j] - 1);
                    }
                }
                index += tmp;
            }

            $t.flat_index(index)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::{index, tensor, Tensor};

    #[test]
    fn tensors__test_scalar_rank_0_tensor() {
        let t = tensor!(i32, [123]);
        assert_eq!(t.dimensions, 1);
        assert_eq!(t.shape, [1]);
        assert_eq!(t.rank, 0);
        assert_eq!(t.value.len(), 1);
        assert_eq!(t.value[0], 123);
    }

    #[test]
    fn tensors__test_row_vector_rank_1_tensor() {
        let v = tensor!([[1, 2, 3]]);
        assert_eq!(v.dimensions, 3);
        assert_eq!(v.shape, [1, 3]);
        assert_eq!(v.rank, 1);
        assert_eq!(v.value.len(), 3);
    }

    #[test]
    fn tensors__test_column_vector_rank_1_tensor() {
        let col_v = tensor!([[1], [2], [3], [4]]);
        assert_eq!(col_v.dimensions, 4);
        assert_eq!(col_v.rank, 1);
        assert_eq!(col_v.shape, [4, 1]);
        assert_eq!(col_v.value.len(), 4);
    }

    #[test]
    fn tensors__test_matrix_rank_2_tensor() {
        let t = Tensor::<f64>::from("[[1,2,3], [3,4,5], [6,7,8]]");
        assert_eq!(t.dimensions, 3);
        assert_eq!(t.shape, [3, 3]);
        assert_eq!(t.rank, 2);
        assert_eq!(t.value, [1., 2., 3., 3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn tensors__test_rank_6_tensor() {
        let t = tensor!([
            [
                [
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ],
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ]
                ],
                [
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ],
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ]
                ]
            ],
            [
                [
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ],
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ]
                ],
                [
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ],
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ]
                ]
            ]
        ]); // shape (2, 2, 2, 2, 4, 4)
        assert_eq!(t.rank, 6);
        assert_eq!(t.dimensions, 4);
        assert_eq!(t.shape, [2, 2, 2, 2, 4, 4]);
        assert_eq!(t.value.len(), 2 * 2 * 2 * 2 * 4 * 4);
    }

    #[test]
    fn tensors__test_rank_4_tensor_indices() {
        let t = tensor!(
            i64,
            [
                [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12],]],
                [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24],]]
            ]
        );

        let shape = t.shape();
        assert_eq!(shape, &[2, 2, 2, 3]);

        let mut n = 1;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    for l in 0..shape[3] {
                        let value = index!(t, i, j, k, l);
                        assert_eq!(value, n);
                        n += 1;
                    }
                }
            }
        }
    }
}
