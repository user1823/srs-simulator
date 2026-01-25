use std::time::Instant;

use pyo3::prelude::*;

mod fsrs_v6;
use fsrs_v6::*;
mod behavior_model;
use behavior_model::*;

use crate::search::simulate;
mod search;

#[pyclass]
struct Lib {
    fsrs: FSRSv6,
}


#[pymethods]
impl Lib {
    #[new]
    fn new(
        weights_vec: Vec<f32>,
        dr_equivalent: f32,
        initial_rating_prob_vec: Vec<f32>,
        initial_cost_vec: Vec<f32>,
        review_rating_prob_given_success_vec: Vec<f32>,
        review_cost_vec: Vec<f32>,
    ) -> PyResult<Self> {
        println!("Rust received:");
        println!("FSRS weights = {:?}", weights_vec);
        println!("DR = {:?}", dr_equivalent);
        println!("initial_rating_prob_vec = {:?}", initial_rating_prob_vec);
        println!("initial_cost_vec = {:?}", initial_cost_vec);
        println!(
            "review_rating_prob_given_success_vec = {:?}",
            review_rating_prob_given_success_vec
        );
        println!("review_cost_vec = {:?}", review_cost_vec);
        if weights_vec.len() != 21 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Expected 21 weights, got {}", weights_vec.len()),
            ));
        }

        let weights: [f32; 21] = weights_vec.try_into().unwrap();
        let initial_rating_prob: [f32; 4] = initial_rating_prob_vec.try_into().unwrap();
        let initial_cost: [f32; 4] = initial_cost_vec.try_into().unwrap();
        let review_rating_prob_given_success: [f32; 3] = review_rating_prob_given_success_vec.try_into().unwrap();
        let review_cost: [f32; 4] = review_cost_vec.try_into().unwrap();
        let sum: f32 = review_rating_prob_given_success.iter().sum();
        let eps = 1e-6;

        assert!(
            (sum - 1.0).abs() < eps,
            "probabilities must sum to 1.0, got {}",
            sum
        );
        let predictor = FSRSv6::new(weights);
        let behavior_model = BehaviorModel::new(initial_rating_prob, initial_cost, review_rating_prob_given_success, review_cost);
        let mut rng = rand::rng();
        let start = Instant::now();
        FSRSv6::test();
        // for i in 0..1 {
        //     let sim_result = simulate(10000.0, 3650.0, &predictor, &behavior_model, &mut rng);
        //     println!("sim result = {:?}", sim_result);
        // }
        let elapsed = start.elapsed();
        println!("Time elapsed: {:.3?}", elapsed); // e.g., 0.500s
        Ok(Lib { fsrs: predictor })
    }
    fn hello(&self) -> PyResult<String> {
        println!("hello world");
        Ok("hello world".to_string())
    }
    fn init(&self, rating: i32) -> PyResult<(f32, (f32, f32))> {
        let state = self.fsrs.first_review(rating);
        let interval = self.fsrs.get_interval(&state, 0.9);
        Ok((interval, state.to_tuple()))
    }
    fn schedule(&self, s: f32, d: f32, rating: i32, elapsed: f32) -> PyResult<(f32, (f32, f32))> {
        let state = FSRSv6State { s: s, d: d };
        let state = self.fsrs.transition(&state, rating, elapsed);
        let interval = self.fsrs.get_interval(&state, 0.9);
        Ok((interval, state.to_tuple()))
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod srs_simulator_rs {
    #[pymodule_export]
    use super::Lib;
}