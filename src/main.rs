use convolutions_rs::convolutions::*;
use convolutions_rs::Padding;
use ndarray::*;

mod utils;
mod Totalistic2D;


fn main() {
    let x = Array::from_shape_vec(
        (1, 4, 4),
        vec![0., 1., 1., 0.,
             1., 0., 1., 1.,
             0., 0., 0., 1.,
             1., 1., 0., 1.]).unwrap();
    let filter = Array::from_shape_vec(
        (1, 1, 3, 3),
        vec![1., 1., 1.,
             1., 0., 1.,
             1., 1., 1.]).unwrap();
    
    let test_vec = Array2::zeros([3, 4]);
    
    let neighbors = conv2d(&filter, None, &x, Padding::Same, 1);
    let _wrapped = utils::wrap_edges(test_vec);
    println!("{}", x);
    println!("{}", neighbors);
    println!("{}", -1 % 2);
}
