use clappers::Clappers;
use ndarray::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_npy::write_npy;
use ndarray_npy::NpzWriter;
use std::fs;
use std::fs::File;
use crossbeam::thread;


mod totalistic_2d;
mod utils;

fn main() {
    let clappers = Clappers::build()
        .set_flags(vec![
            "h|help",
            "transient",
            "save-rules",
            "save-all",
        ])
        .set_singles(vec![
            "r|random",
            "f|file",
            "t|threads",
        ])
        .parse();

    if clappers.get_flag("help") {
        println!("
            usage: cargo run {{-r num_states|-f rule_file}} <grid> <steps> <runs>

            Arguments:
                -h,--help        Print this help
                -f,--file        path to a file containing the rule table for simulation
                -r,--random      generate a random rule trable for simulations
                -t,--threads     number of threads to use in performing computation
                --transient      simulate for <steps> and then return the transient
                --save-all       save all states the system passes through
        ");
    }

    // variables we will be assigning from CLI
    let mut rule_table: Array2<usize> = arr2(&[[0], [0]]);
    let rule_path = clappers.get_single("f");
    
    // handle random rule table
    let rand_in = clappers.get_single("r");
    if !rand_in.is_empty() {
        let rand_in = match rand_in.parse::<usize>() {
            Ok(number) => number,
            Err(error) => panic!("Bad value passed to random option. Original error: {:?}", error),
        };
        rule_table = Array2::random((rand_in, 9), Uniform::from(0..rand_in));
        
    }
    
    // multi-threading
    let num_threads_input = clappers.get_single("t");
    let mut num_threads: usize = 1;
    
    if !num_threads_input.is_empty() {
        num_threads = match num_threads_input.parse::<usize>() {
            Ok(threads) => threads,
            Err(e) => panic!("Bad value passed to threads option. Original error: {:?}", e),
        };
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
        rule_table = Array2::<usize>::from_shape_vec((nlines, 9), arr_vec).unwrap();
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

    // the actual simulations all occur below this line


    // initialize model and run according to whether or not the transient flag is passed
    let num_states = rule_table.nrows();
    
    let model = totalistic_2d::Totalistic2D {
        on_state: (num_states - 1) as f32,
        thresholds: rule_table,
    };

    // indices for multithreading
    let chunks = (runs as f64 / num_threads as f64).floor() as usize;
    let run_rem = runs % num_threads;
    
    // save transient length only
    if clappers.get_flag("transient") {
        if clappers.get_flag("save-rules") {
            let save_rule_table = model.thresholds.mapv(|elem| elem as u32);
            println!("saving rules");
            write_npy("data/rule_table.npy", &save_rule_table).unwrap();
        }
        
        let mut model_transients = Vec::with_capacity(runs);

        // we chunk the runs by the number of threads for parallelizations
        for _chunk in 0..chunks {
            // define parallel scope
            let s = thread::scope(|s| {

                // collects values from each thread
                let mut handles: Vec<thread::ScopedJoinHandle<usize>> = Vec::new();
                for _run in 0..num_threads {

                    // heres where we perform the actual simulation
                    handles.push(s.spawn(|_| {

                        // this is all in a scoped thread
                        let initial_cond = Array2::<usize>::random((grid, grid), Uniform::from(0..num_states));
                        let t_out = model.simulate_transient(utils::int_to_float(initial_cond), steps);
                        t_out.length
                    }));
                }
                
                // join the handles into a single vector for the scope
                let mut chunk_transients = Vec::with_capacity(num_threads);
                for handle in handles {
                    chunk_transients.push(handle.join().unwrap());
                }
                // return the scope transients
                chunk_transients
            });

            // back in main thread. extend the transient vec by the scope vector
            model_transients.append(&mut s.unwrap());
        }

        // we also have to handle the remaining runs
        let s = thread::scope(|s| {
            let mut handles: Vec<thread::ScopedJoinHandle<usize>> = Vec::new();
            for _run in 0..run_rem {
                handles.push(s.spawn(|_| {
                    let initial_cond = Array2::<usize>::random((grid, grid), Uniform::from(0..num_states));
                    let t_out = model.simulate_transient(utils::int_to_float(initial_cond), steps);
                    t_out.length
                }));
            }
            // again unpack the handles
            let mut remain_transients = Vec::with_capacity(run_rem);
            for handle in handles {
                remain_transients.push(handle.join().unwrap());
            }

            // return remaining transients
            remain_transients
        });

        // main thread. extend the transient vec
        model_transients.append(&mut s.unwrap());

        // we need to convert all values to a specific type to save npy
        let save_transients: Vec<u32>;
        save_transients = model_transients.iter().map(|elem| *elem as u32).collect();
        write_npy("data/transients.npy", &arr1(&save_transients)).unwrap();
    
    // save entire time series
    } else {
        if clappers.get_flag("save-rules") {
            let save_rule_table = model.thresholds.mapv(|elem| elem as u32);
            println!("saving rules");
            write_npy("data/rule_table.npy", &save_rule_table).unwrap();
        }
        let mut npz = NpzWriter::new(File::create("data/time_series.npz").unwrap());
        for run in 0..runs {
            let initial_cond = Array2::random((grid, grid),
                                                             Uniform::from(0..num_states));
            let time_series = model.simulate(utils::int_to_float(initial_cond), steps);
            if clappers.get_flag("save-all") {
                let save_time_series = time_series.mapv(|elem| elem as u32);
                npz.add_array(run.to_string(), &save_time_series).unwrap();
            }
        }
        npz.finish().unwrap();
    }
}
