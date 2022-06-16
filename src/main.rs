use clappers::Clappers;
use ndarray::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::fs;

mod totalistic_2d;
mod utils;


fn main() {
    let clappers = Clappers::build()
        .set_flags(vec![
            "h|help"
        ])
        .set_singles(vec![
            "r|random",
            "f|file",
        ])
        .parse();

    if clappers.get_flag("help") {
        println!("
            usage: cargo run {{-r num_states|-f rule_file}} <grid> <steps> <runs>

            Arguments:
                -h,--help        Print this help
                -f,--file        path to a file containing the rule table for simulation
                -r,--random      generate a random rule trable for simulations
        ");
    }

    // variables we will be assigning from CLI
    let mut rule_table = arr2(&[[0], [0]]);
    let rule_path = clappers.get_single("f");

    // handle random rule table
    let rand_in = clappers.get_single("r");
    if !rand_in.is_empty() {
        let rand_in = match rand_in.parse::<usize>() {
            Ok(number) => number,
            Err(error) => panic!("Bad value passed to random option. Original error: {:?}", error),
        };
        rule_table = Array2::random((rand_in, 9), Uniform::from(0..rand_in));
    
    // handle file rule table. must be in format
    // #,#,#,#,#,#,#,#,#,#
    // #,#,#,#,#,#,#,#,#,#
    } else if !rule_path.is_empty() {
        let f_str = match fs::read_to_string(rule_path.clone()) {
            Ok(text) => text,
            Err(error) => panic!("File not found! Original error: {:?}", error),
        };
        let mut arr_vec: Vec<usize> = Vec::new();
        // probably a better way to do that
        let flines: Vec<&str> = f_str.lines().collect();
        let nlines = flines.len();
        for line in f_str.lines() {
            let sep = line.split(",");
            for char in sep {
                let ichar = match char.parse::<usize>() {
                    Ok(number) => number,
                    Err(error) => panic!("Bad symbol in {:?}!\n{:?}", rule_path, error),
                };
                arr_vec.push(ichar);
            }
        }
        rule_table = Array2::from_shape_vec((nlines, 9), arr_vec).unwrap();
    }

    // get leftover variables from command line
    let grid_steps_runs = clappers.get_leftovers();
    let grid = match grid_steps_runs[0].parse::<usize>() {
            Ok(number) => number,
            Err(error) => panic!("Grid size must be an integer!\n{:?}", error),
    };
    let steps = match grid_steps_runs[1].parse::<usize>() {
            Ok(number) => number,
            Err(error) => panic!("Number of steps must be an integer!\n{:?}", error),
    };
    let runs = match grid_steps_runs[2].parse::<usize>() {
            Ok(number) => number,
            Err(error) => panic!("Number of runs must be an integer!\n{:?}", error),
    };

    // initialize and run the model
    let num_states = rule_table.nrows();
    
    let model = totalistic_2d::Totalistic2D {
        on_state: (num_states - 1) as f32,
        thresholds: rule_table,
    };
    for run in 0..runs {
        let initial_conditions = Array2::<usize>::random((grid, grid),
                                                         Uniform::from(0..num_states));
        let _ = model.simulate(utils::int_to_float(initial_conditions), steps);
    }

    println!("done");
}
