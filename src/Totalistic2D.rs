use ndarray::{Array2, Array3, Axis, arr2, s};
use crate::utils;

#[derive(Debug)]
struct Totalistic2D {
    // the state that we count for our neighbors
    on_state: f32,
    // this is a (states x 9) array that relates current state to next
    // state based on the number of neighbors. for example the array
    // [[0, 0, 1, 0, 1, 1, 1, 1, 1],
    //  [0, 0, 0, 0, 0, 0, 1, 1, 1]]
    // would tell you that a cell in state 0 with 4 neighbors, index [[0, 4]],
    // will be in state 0 in the next time step
    thresholds: ndarray::Array2<usize>,
}



impl Totalistic2D {
    // core function for updating the state of the system
    fn step(&self, grid: &Array2<f32>) -> Array2<f32> {
        let mut next_grid = Array2::<f32>::zeros(grid.raw_dim());
        let on_grid = utils::filter_grid(&grid, self.on_state);
        let padded_grid = utils::wrap_edges(on_grid);
        let neighbor_kernel = arr2(&[[1., 1., 1.],
                                     [1., 0., 1.],
                                     [1., 1., 1.]]);
        let padded_neighbors = utils::flat_conv2d(padded_grid, neighbor_kernel);
        let neighbors = padded_neighbors.slice(s![1..-1, 1..-1]);
        for ri in 0..neighbors.nrows() {
            for ci in 0..neighbors.ncols() {
                // need to use these values for indexing. s for state
                // all this type casting might be a problem??
                let int_s = grid[[ri, ci]] as usize;
                let int_n = neighbors[[ri, ci]] as usize;

                next_grid[[ri, ci]] = self.thresholds[[int_s, int_n]] as f32;                
            }
        }
        next_grid
    }

    // the function we will call in main to simulat a number of steps
    pub fn simulate(&self, initial: Array2<f32>, steps: usize) -> Array3<f32> {
        let shape = initial.dim();
        let mut next_state = initial.clone();
        let mut out_arr = initial.into_shape((1, shape.0, shape.1)).unwrap();
        for _ in 1..steps {
            let state = &next_state;
            next_state = self.step(state);
            out_arr.push(Axis(0), next_state.view()).unwrap();
        }
        out_arr
    }
}

#[cfg(test)]
mod test_totalistic2d {
    use super::*;
    use ndarray::arr3;

    #[test]
    fn test_step() {
        let test_model = Totalistic2D {
            on_state: 2.,
            thresholds: arr2(&[[2, 0, 1, 1, 1, 1, 2, 2, 2],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 2, 2, 2, 2]]),
        };
        let in_grid = arr2(&[[1., 0., 0.],
                             [1., 0., 1.],
                             [0., 0., 1.]]);
        let out_grid = arr2(&[[0., 2., 2.],
                              [0., 2., 0.],
                              [2., 2., 0.]]);
        assert_eq!(out_grid, test_model.step(&in_grid))
    }

    #[test]
    fn test_simulate() {
        let test_model = Totalistic2D {
            on_state: 2.,
            // neighbors        0  1  2  3  4  5  6  7  8
            thresholds: arr2(&[[2, 0, 1, 1, 1, 1, 2, 2, 2],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 2, 2, 2, 2]]),
        };
        let in_grid = arr2(&[[2., 1., 2.],
                             [2., 0., 1.],
                             [0., 1., 1.]]);
        let out_grid = arr3(&[[[2., 1., 2.],
                               [2., 0., 1.],
                               [0., 1., 1.]],

                              [[0., 0., 0.],
                               [0., 1., 0.],
                               [1., 0., 0.]],

                              [[2., 2., 2.],
                               [2., 0., 2.],
                               [0., 2., 2.]],

                              [[2., 2., 2.],
                               [2., 2., 2.],
                               [2., 2., 2.]],

                              [[2., 2., 2.],
                               [2., 2., 2.],
                               [2., 2., 2.]],

                              [[2., 2., 2.],
                               [2., 2., 2.],
                               [2., 2., 2.]]]);
        assert_eq!(out_grid, test_model.simulate(in_grid, 6))
    }
}