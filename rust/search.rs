use rand::Rng;

use crate::{behavior_model::BehaviorModel, fsrs_adr::FSRSADRGenerator, fsrs_v6::{FSRSv6, FSRSv6State}};
use crate::{fsrs_adr::FSRSADR};
extern crate rand;

#[derive(Debug)]
pub struct SimResult {
    pub total_average_memorized: f64,
    pub total_cost: f64,
}
impl SimResult {
    fn efficiency(&self) -> f64 {
        self.total_average_memorized / self.total_cost
    }
}

fn get_proportion(t: f32, limit_t: f32, end_t: f32) -> f32 {
    if t < limit_t {
        1.0
    } else if t == end_t {
        0.0
    } else {
        1.0 - (t - limit_t) / (end_t - limit_t)
    }
}

fn simulate_review_card<T: Rng>(
    start_weight: f32,
    start_t: f32,
    limit_t: f32,
    end_t: f32,
    start_state: FSRSv6State,
    predictor: &FSRSv6,
    adr_model: &FSRSADR,
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
        let dr = adr_model.get_dr(state.s, state.d);
        let (interval, r) = predictor.schedule(&state, dr);
        let review_day = f32::min(t + interval, end_t);
        let time_existing_in_memory = review_day - t;
        let review_day_proportion = get_proportion(review_day, limit_t, end_t);
        // total memorized is the same between all rating options
        accum_sim_result.total_average_memorized += {
            if t < limit_t && limit_t < review_day {
                let volume = 
                    predictor.forgetting_curve_volume(&state, limit_t - t)
                    + predictor.forgetting_curve_volume_weighted(
                        &state, 
                        limit_t - t, 
                        review_day - t,
                        1.0,
                        get_proportion(review_day, limit_t, end_t));

                weight * volume
            } else if t < limit_t {
                weight * predictor.forgetting_curve_volume(&state, time_existing_in_memory)
            } else {
                weight * predictor.forgetting_curve_volume_weighted(
                    &state, 
                    0.0, 
                    time_existing_in_memory, 
                    get_proportion(t, limit_t, end_t),
                    review_day_proportion,
                )
            }
        } as f64;

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
                accum_sim_result.total_cost += (next_weight * review_day_proportion * behavior_model.review_cost(rating_idx)) as f64;

                let split_result = simulate_review_card(
                    next_weight,
                    review_day,
                    limit_t,
                    end_t,
                    predictor.transition(&state, rating, interval),
                    &predictor,
                    &adr_model,
                    &behavior_model,
                    rng,
                );
                accum_sim_result.total_average_memorized += split_result.total_average_memorized;
                accum_sim_result.total_cost += split_result.total_cost;
            }
        }

        let next_weight = weight * cont_prob;
        accum_sim_result.total_cost += (weight * review_day_proportion * cont_prob * behavior_model.review_cost(cont_rating_idx)) as f64;

        // Prepare state for the next iteration
        weight = next_weight;
        t = review_day;
        state = predictor.transition(&state, cont_rating_idx as i32 + 1, interval);
        l += 1;
    }
    // if start_weight > 100.0 {
    //     println!("start {} end {} t = {} len = {}", start_weight, weight, start_t, l);
    // }
    accum_sim_result
}

pub fn simulate<T: Rng>(
    weight: f32,
    deck_size: i32,
    new_cards_per_day: i32,
    end_t: f32,
    predictor: &FSRSv6,
    adr_model: &FSRSADR,
    behavior_model: &BehaviorModel,
    rng: &mut T,
) -> SimResult {
    let learn_days = deck_size as f32 / new_cards_per_day as f32;
    let limit_t = f32::max(0.0, end_t - learn_days);
    let mut accum_sim_result = SimResult { total_average_memorized: 0.0, total_cost: 0.0 };
    for rating_idx in 0..4 {
        let rating = rating_idx as i32 + 1;
        let p = behavior_model.initial_rating_prob(rating_idx);
        accum_sim_result.total_cost += (p * weight * behavior_model.initial_cost(rating_idx)) as f64;
        let init_state = predictor.first_review(rating);
        let split_result = simulate_review_card(p * weight, 0.0, limit_t, end_t, init_state, &predictor, &adr_model, &behavior_model, rng);
        accum_sim_result.total_average_memorized += split_result.total_average_memorized;
        accum_sim_result.total_cost += split_result.total_cost;
    }
    accum_sim_result
}

pub fn simulated_annealing(
    dr_equivalent: f32,
    deck_size: i32,
    new_cards_per_day: i32,
    days: i32,
    predictor: &FSRSv6,
    behavior_model: &BehaviorModel,
) -> FSRSADR {
    let generator = FSRSADRGenerator {};
    let mut rng = rand::rng();
    let mut best_adr = FSRSADR::fixed_dr(dr_equivalent);
    let baseline_result = simulate(10000.0, deck_size, new_cards_per_day, days as f32, &predictor, &best_adr, &behavior_model, &mut rng);
    let baseline_memorized = baseline_result.total_average_memorized;
    let mut best_score = baseline_result.efficiency();
    let mut cur_adr = best_adr.clone();
    let mut cur_score = best_score;
    let temp_initial: f64 = 0.2;
    let temp_final: f64 = 0.003;
    let mut fail_counter = 0;
    let initial_fail_counter_target = 4;
    let mut fail_counter_target = initial_fail_counter_target;
    let n_iterations = 3000;
    for it in 0..n_iterations {
        let sample = generator.suggest(&cur_adr, &mut rng);
        let temp_choice: f64 = rng.random();
        let mut eval = |model: &FSRSADR| -> SimResult {
            simulate(10000.0, deck_size, new_cards_per_day, days as f32, &predictor, &model, &behavior_model, &mut rng)
        };

        // let result = simulate(10000.0, deck_size, new_cards_per_day, days as f32, &predictor, &sample, &behavior_model, &mut rng);
        let result = eval(&sample);
        if result.total_average_memorized < baseline_memorized {
            continue
        } 
        println!("it: {}, memorized: {:.3}, eff: {:.3}", it, result.total_average_memorized, result.efficiency());
        let score = result.efficiency();
        let temp = temp_initial * (temp_final / temp_initial).powf(it as f64 / n_iterations as f64);
        if score > best_score {
            let verify_score = eval(&sample).efficiency();
            if verify_score > best_score {
                println!("---- Global Best ---- first: {:.3} second: {:.3}", score, verify_score);
                best_score = verify_score;
                best_adr = sample.clone();
                println!("Model = {:?}", best_adr);
            }
        }
        if score > cur_score {
            println!("Local best.");
            cur_score = score;
            cur_adr = sample;
            fail_counter = 0;
            fail_counter_target = initial_fail_counter_target;
        } else if temp_choice < ((score - cur_score) / temp).exp() {
            println!("Temp transition {} {} {}", score, cur_score, temp);
            cur_score = score;
            cur_adr = sample;
            fail_counter = 0;
            fail_counter_target = initial_fail_counter_target;
        } else {
            println!("Do nothing.");
            fail_counter += 1;
            if fail_counter == fail_counter_target {
                let prev_score = cur_score;
                cur_score = eval(&cur_adr).efficiency();
                fail_counter = 0;
                fail_counter_target = 2 * fail_counter_target;
                println!("Rerolling... {:.3} -> {:.3}", prev_score, cur_score);
            }
        }
    }
    println!("Initial score: {}, Final score: {}", baseline_result.efficiency(), best_score);
    // let verify = simulate(10000.0, deck_size, new_cards_per_day, days as f32, &predictor, &best_adr, &behavior_model, &mut rng);
    let verify = simulate(10000.0, deck_size, new_cards_per_day, days as f32, &predictor, &best_adr, &behavior_model, &mut rng);
    println!("Best score repeated: {:.3}", verify.efficiency());
    println!("Best adr: {:?}", best_adr);
    println!("Cur adr: {:?}", cur_adr);
    best_adr
}