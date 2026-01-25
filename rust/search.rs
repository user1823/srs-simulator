use rand::Rng;

use crate::{behavior_model::BehaviorModel, fsrs_v6::{FSRSv6, FSRSv6State}};
extern crate rand;

#[derive(Debug)]
pub struct SimResult {
    pub total_average_memorized: f32,
    pub total_cost: f32,
}

fn simulate_review_card<T: Rng>(
    start_weight: f32,
    start_t: f32,
    end_t: f32,
    start_state: FSRSv6State,
    predictor: &FSRSv6,
    behavior_model: &BehaviorModel,
    rng: &mut T,
) -> SimResult {
    let mut weight = start_weight;
    let mut t = start_t;
    let mut state = start_state;
    let mut accum_sim_result = SimResult { total_average_memorized: 0.0, total_cost: 0.0 };
    let mut split = weight > 1.0;
    let mut l = 0;
    loop {
        let dr = 0.9;
        let (interval, r) = predictor.schedule(&state, dr);
        let review_day = t + interval;
        // total memorized is the same between all rating options
        accum_sim_result.total_average_memorized += {
            let time_existing_in_memory = f32::min(review_day, end_t) - t;
            weight * predictor.forgetting_curve_volume(&state, time_existing_in_memory)
        };

        if review_day >= end_t {
            break
        }
        split = split && weight > 1.0;  // Maybe the compiler can pull some magic when written in this form
        let probs = behavior_model.review_rating_prob_dist(r);
        let (cont_rating_idx, cont_prob) = 
            if split {
                let mut max_idx = 0usize;
                let mut max_value = probs[0];
                for i in 1..probs.len() {
                    if probs[i] > max_value {
                        max_value = probs[i];
                        max_idx = i;
                    }
                }
                (max_idx, max_value)
            } else {
                (behavior_model.sample_review_rating_idx(r, rng), 1.0)
            };


        if split {
            for i in 0..probs.len() as i32 {
                if i as usize == cont_rating_idx {
                    continue
                }
                let rating_idx = i as usize;
                let rating: i32 = i + 1;
                let next_weight = weight * probs[rating_idx];
                accum_sim_result.total_cost += next_weight * behavior_model.review_cost(rating_idx);

                let split_result = simulate_review_card(
                    next_weight,
                    review_day,
                    end_t,
                    predictor.transition(&state, rating, interval),
                    &predictor,
                    &behavior_model,
                    rng,
                );
                accum_sim_result.total_average_memorized += split_result.total_average_memorized;
                accum_sim_result.total_cost += split_result.total_cost;
            }
        }

        let next_weight = weight * cont_prob;
        accum_sim_result.total_cost += weight * cont_prob * behavior_model.review_cost(cont_rating_idx);

        // Prepare state for the next iteration
        weight = next_weight;
        t = review_day;
        state = predictor.transition(&state, cont_rating_idx as i32 + 1, interval);
        l += 1;
    }
    if start_weight > 100.0 {
        println!("start {} end {} t = {} len = {}", start_weight, weight, start_t, l);
    }
    accum_sim_result
}

pub fn simulate<T: Rng>(
    weight: f32,
    end_t: f32,
    predictor: &FSRSv6,
    behavior_model: &BehaviorModel,
    rng: &mut T,
) -> SimResult {
    let mut accum_sim_result = SimResult { total_average_memorized: 0.0, total_cost: 0.0 };
    for rating_idx in 0..4 {
        let rating = rating_idx as i32 + 1;
        let p = behavior_model.initial_rating_prob(rating_idx);
        accum_sim_result.total_cost += weight * behavior_model.initial_cost(rating_idx);
        let init_state = predictor.first_review(rating);
        let split_result = simulate_review_card(p * weight, 0.0, end_t, init_state, &predictor, &behavior_model, rng);
        accum_sim_result.total_average_memorized += split_result.total_average_memorized;
        accum_sim_result.total_cost += split_result.total_cost;
    }
    accum_sim_result
}

pub fn simulated_annealing(
    predictor: &FSRSv6,
    behavior_model: &BehaviorModel,
) {

}