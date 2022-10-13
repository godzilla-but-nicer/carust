use ndarray::*;
use convolutions_rs::Padding;
use convolutions_rs::convolutions::conv2d;

// convert an array of ints to an array of floats
pub fn int_to_float(a: ndarray::Array2<usize>) -> ndarray::Array2<f32> {
    let mut float_array = Array2::zeros(a.dim());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            float_array[[i, j]] = a[[i, j]] as f32;
        }
    }
    float_array
}

// filter a 2d array such that elements == keep_state are 1. and all else are 0.
pub fn filter_grid(a: &ndarray::Array2<f32>, keep_state: f32) -> ndarray::Array2<f32> {
    let mut out_arr = Array2::<f32>::zeros(a.raw_dim());
    for ri in 0..a.nrows() {
        for ci in 0..a.ncols() {
            if a[[ri, ci]] == keep_state {
                out_arr[[ri, ci]] = 1.;
            }
        }
    }
    out_arr
}

// pad a 2d array such that the inner elements appear to have
// toroidal boundry conditions
// needs work probably. assign_to() method for slice will make this easy
pub fn wrap_edges(a: ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    // i need to use the last index a lot
    let arows = a.nrows();
    let acols = a.ncols();

    
    // top and bottom rows are different than the others
    let mut top_row = vec![0.; acols+2];
    top_row[0] = a[[arows-1, acols-1]];
    top_row[acols + 1] = a[[arows-1, 0]];
    // iterate to fill the middle of the row
    for i in 1..(acols+1) {
        top_row[i] = a[[arows-1, i-1]];
    }
    let mut out_grid = Array::from_shape_vec((1, acols+2), top_row).unwrap();

    // the same proceedure for the bottom
    let mut bot_row = vec![0.; acols+2];
    bot_row[0] = a[[0, acols-1]];
    bot_row[acols+1] = a[[0, 0]];
    for i in 1..(acols+1) {
        bot_row[i] = a[[0, i-1]];
    }

    // middle rows are all similar
    for ri in 0..arows {
        let mut out_row = vec![0.; acols + 2];
        // each row then looks similar to the above
        out_row[0] = a[[ri, (acols-1)]];
        out_row[acols+1] = a[[ri, 0]];
        for ci in 1..(acols+1) {
            out_row[ci] = a[[ri, ci-1]];
        }
        out_grid.push_row(ArrayView::from(&out_row)).unwrap();
    }

    out_grid.push_row(ArrayView::from(&bot_row)).unwrap();
    out_grid
}

// convolution library requires an Array3
pub fn flat_conv2d(a: ndarray::Array2<f32>, kernel: ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    // convolution_rs requires channels in first dimension for the array
    let a_shape = (1, a.nrows(), a.ncols());
    let flat_shape = (a.nrows(), a.ncols());
    let a_conv = a.into_shape(a_shape).unwrap();

    // also requires channels in, channels out for the kernel
    let k_shape = (1, 1, kernel.nrows(), kernel.ncols());
    let k_conv = kernel.into_shape(k_shape)
                       .unwrap();
    
    let filtered = conv2d(&k_conv, None, &a_conv, Padding::Same, 1);
    let flat_filtered = filtered.into_shape(flat_shape).unwrap();
    flat_filtered
}

#[cfg(test)]
mod test_utils {
    use super::*;
    #[test]
    fn test_int_to_float() {
        let int_grid =   array![[1, 2, 1, 3],
                                [2, 1, 2, 1],
                                [1, 2, 3, 4]];
        let float_grid = array![[1., 2., 1., 3.],
                                [2., 1., 2., 1.],
                                [1., 2., 3., 4.]];
        assert_eq!(float_grid, int_to_float(int_grid))
    }
    #[test]
    fn test_filter_grid() {
        let unfiltered_grid =   array![[1., 2., 1., 3.],
                                       [2., 1., 2., 1.],
                                       [1., 2., 3., 4.]];
        let filtered_grid =   array![[0., 1., 0., 0.],
                                     [1., 0., 1., 0.],
                                     [0., 1., 0., 0.]];
        assert_eq!(filtered_grid, filter_grid(&unfiltered_grid, 2.))
    }
    #[test]
    fn test_wrap_edges() {
        let unwrapped_grid =   array![[1., 0., 1., 0.],
                                      [0., 1., 0., 1.],
                                      [1., 0., 0., 0.]];

        let wrapped_grid = array![[0., 1., 0., 0., 0., 1.],
                                  [0., 1., 0., 1., 0., 1.],
                                  [1., 0., 1., 0., 1., 0.],
                                  [0., 1., 0., 0., 0., 1.],
                                  [0., 1., 0., 1., 0., 1.]];
        let output_grid = wrap_edges(unwrapped_grid);
        println!("{}", output_grid);
        assert_eq!(wrapped_grid, output_grid)
    }
    #[test]
    fn test_flat_conv2d() {
        let unwrapped_grid = array![[1., 0., 1., 0.],
                                    [0., 1., 0., 1.],
                                    [1., 0., 0., 0.]];
        let neighbors = array![[1., 3., 2., 2.],
                               [3., 3., 3., 1.],
                               [1., 2., 2., 1.]];
        let filter = array![[1., 1., 1.],
                            [1., 0., 1.],
                            [1., 1., 1.]];
        let conv_out = flat_conv2d(unwrapped_grid, filter);
        assert_eq!(conv_out, neighbors)
    }
}