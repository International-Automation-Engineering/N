use tensors::{index, tensor, Tensor};

fn main() {
    let t0 = tensor!([[[[1, 2, 3], [4, 5, 6]]], [[[7, 8, 9], [10, 11, 12]]]]);
    println!("{:?}", t0);
    let v0 = index!(t0, 0, 0, 1, 2);
    let v1 = index!(t0, 1, 0, 1, 2);
    println!("{} {}", v0, v1);
}
