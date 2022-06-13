#[derive(Debug)]
struct 2DTotalistic {
    states: int,
    thresholds: Vec,
}

impl 2DTotalistic {
    fn step(&self, grid: Vec<Vec<int>>) -> Vec<Vec<int>> {
        new_grid: Vec<Vec<int>> = vec![vec![0; grid.len()]; grid.len()];
    }
}