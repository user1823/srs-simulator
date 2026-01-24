use pyo3::prelude::*;

mod fsrs_v6;
use fsrs_v6::*;

#[pyclass]
struct Lib {
    fsrs: FSRSv6
}


#[pymethods]
impl Lib {
    #[new]
    fn new(weights_vec: Vec<f32>) -> PyResult<Self> {
        println!("Weights vec32 = {:?}", weights_vec);
        // Ensure the Python sequence has exactly 21 elements
        if weights_vec.len() != 21 {
               // Correct usage: return Err(PyErr)
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Expected 21 weights, got {}", weights_vec.len()),
            ));
        }

        // Convert Vec<f32> -> [f32; 21]
        let weights: [f32; 21] = weights_vec.try_into().unwrap();
        Ok(Lib { fsrs: FSRSv6::new(weights) })
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
        let state = self.fsrs.transition(state, rating, elapsed);
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