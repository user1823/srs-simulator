use rand::Rng;

use crate::{behavior_model::BehaviorModel, fsrs_adr::FSRSADRGenerator, fsrs_v6::{FSRSv6, FSRSv6State}};
use crate::{fsrs_adr::FSRSADR};
extern crate rand;

#[derive(Debug, Clone)]
pub struct SimResult {
    pub total_average_memorized: f64,
    pub total_cost: f64,
    pub weight: f32,
    pub days: f32,
    pub total_iters: i32,
}
impl SimResult {
    fn efficiency_confidence(&self) -> f64 {
        self.total_average_memorized / self.total_cost - 1e1 * (1.0 / self.total_iters as f64).sqrt()
    }
    fn efficiency(&self) -> f64 {
        self.total_average_memorized / self.total_cost
    }
    fn memorized(&self) -> f64 {
        self.total_average_memorized / (self.weight as f64 * self.days as f64)
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
    let mut accum_sim_result = SimResult { total_average_memorized: 0.0, total_cost: 0.0, weight: start_weight, days: end_t - start_t, total_iters: 0 };
    let mut split = weight > 1.0;
    let mut l = 0;
    let mut monte_carlo_len = 0;
    let mut monte_carlo_prune_len = 0;
    let mut monte_carlo_start_t = 0.0;
    let mut monte_carlo_memorized = 0.0;
    let mut monte_carlo_cost = 0.0;
    let mut can_prune = false;

    loop {
        let dr = adr_model.get_dr(state.s, state.d);
        let (interval, r) = predictor.schedule(&state, dr);
        let review_day = f32::min(t + interval, end_t);
        let time_existing_in_memory = review_day - t;
        let review_day_proportion = get_proportion(review_day, limit_t, end_t);
        if !split {
            // Early return of random simulations
            let monte_carlo_elapsed_t = t - monte_carlo_start_t;
            if can_prune && monte_carlo_len > monte_carlo_prune_len && monte_carlo_start_t + 2.0 * monte_carlo_elapsed_t < end_t {
                let memorized_vol_per_day = monte_carlo_memorized / monte_carlo_elapsed_t;
                let cost_per_day = monte_carlo_cost / monte_carlo_elapsed_t;
                let remaining_day_volume = {
                    let rect = (limit_t - t).max(0.0);
                    let triangle = 0.5 * get_proportion(t, limit_t, end_t) * (end_t - f32::max(limit_t, t));
                    rect + triangle
                };
                let est_memorized = weight * memorized_vol_per_day * remaining_day_volume;
                let est_cost = weight * cost_per_day * remaining_day_volume;
                accum_sim_result.total_average_memorized += est_memorized as f64;
                accum_sim_result.total_cost += est_cost as f64;

                return accum_sim_result;
            }
        }

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
        if split && weight <= 1.0 {
            // Setup monte carlo
            monte_carlo_start_t = t;
            split = false;
            can_prune = rng.random_bool(0.99);
            monte_carlo_prune_len = rng.random_range(32..64);
        }
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
        if !split && can_prune {
            monte_carlo_len += 1;
            monte_carlo_memorized += predictor.forgetting_curve_volume(&state, time_existing_in_memory);
            monte_carlo_cost += behavior_model.review_cost(cont_rating_idx);
        }
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
                accum_sim_result.total_iters += split_result.total_iters;
            }
        }

        let next_weight = weight * cont_prob;
        accum_sim_result.total_cost += (weight * review_day_proportion * cont_prob * behavior_model.review_cost(cont_rating_idx)) as f64;
        accum_sim_result.total_iters += 1;

        // Prepare state for the next iteration
        weight = next_weight;
        t = review_day;
        state = predictor.transition(&state, cont_rating_idx as i32 + 1, interval);
        l += 1;
    }
    // if l > 1000 {
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
    let mut accum_sim_result = SimResult { total_average_memorized: 0.0, total_cost: 0.0, weight: weight, days: end_t, total_iters: 0 };
    for rating_idx in 0..4 {
        let rating = rating_idx as i32 + 1;
        let p = behavior_model.initial_rating_prob(rating_idx);
        accum_sim_result.total_cost += (p * weight * behavior_model.initial_cost(rating_idx)) as f64;
        let init_state = predictor.first_review(rating);
        let split_result = simulate_review_card(p * weight, 0.0, limit_t, end_t, init_state, &predictor, &adr_model, &behavior_model, rng);
        accum_sim_result.total_average_memorized += split_result.total_average_memorized;
        accum_sim_result.total_cost += split_result.total_cost;
        accum_sim_result.total_iters += split_result.total_iters;
    }
    accum_sim_result
}

fn get_memorized_penalty(result: &SimResult, it_ratio: f32, baseline_memorized: f32) -> f32 {
    let memorized_multi = 1e4;
    memorized_multi * it_ratio * f32::max(0.0, baseline_memorized as f32 - result.memorized() as f32)
}

pub fn simulated_annealing(
    dr_equivalent: f32,
    deck_size: i32,
    new_cards_per_day: i32,
    days: i32,
    predictor: &FSRSv6,
    behavior_model: &BehaviorModel,
) -> FSRSADR {
    let mut generator = FSRSADRGenerator::new();
    let mut rng = rand::rng();
    let baseline_adr = FSRSADR::fixed_dr(dr_equivalent);
    let mut best_adr = baseline_adr.clone();
    let baseline_result = simulate(30000.0, deck_size, new_cards_per_day, days as f32, &predictor, &best_adr, &behavior_model, &mut rng);
    let baseline_memorized = baseline_result.memorized() as f32;
    let mut best_score = baseline_result.efficiency_confidence() as f32;
    let mut best_result = baseline_result.clone();
    let mut cur_adr = best_adr.clone();
    let mut cur_score = best_score;
    let mut cur_result = baseline_result.clone();
    let temp_initial: f32 = 0.1;
    let temp_final: f32 = 0.004;
    let target_iters_initial: f32 = 10000.0;
    let target_iters_final: f32 = 600000.0;
    let mut fail_counter = 0;
    let initial_fail_counter_target = 4;
    let mut fail_counter_target = initial_fail_counter_target;
    let mut simulate_weight = 10.0;
    // return best_adr;
    let n_iterations = 2000;
    for it in 0..n_iterations {
        let it_ratio = it as f32 / n_iterations as f32;
        let sample = generator.suggest(&cur_adr, &mut rng);
        let temp_choice: f32 = rng.random();
        let simulate_weight_copy = simulate_weight;
        let mut eval = |model: &FSRSADR| -> SimResult {
            simulate(simulate_weight_copy, deck_size, new_cards_per_day, days as f32, &predictor, &model, &behavior_model, &mut rng)
        };

        let result = eval(&sample);
        let target_iters = target_iters_initial * (target_iters_final / target_iters_initial).powf(it_ratio);
        if result.total_iters as f32 > target_iters {
            simulate_weight = (simulate_weight / 1.1).max(1.0);
        } else {
            simulate_weight = (simulate_weight * 1.1).min(100000.0);
        }

        let memorized_penalty = get_memorized_penalty(&result, it_ratio, baseline_memorized);
        let temp = temp_initial * (temp_final / temp_initial).powf(it_ratio);
        println!("it: {}, temp: {:.3}, memorized: {:.3}, eff: {:.3}, eff_conf: {:.3}, iters: {} weight: {:.1}, eff_conf_penalty: {:.3}, memorized penalty: {:.3}", it, temp, result.memorized(), result.efficiency(), result.efficiency_confidence(), result.total_iters, simulate_weight, -result.efficiency_confidence() + result.efficiency(), memorized_penalty);
        let score = result.efficiency_confidence() as f32 - memorized_penalty;

        if score > cur_score {
            println!("Local best.");
            cur_score = score;
            cur_adr = sample;
            cur_result = result;
            fail_counter = 0;
            fail_counter_target = initial_fail_counter_target;
        } else if temp_choice < ((score - cur_score) / temp).exp() {
            println!("Temp transition {:.3} {:.3} {:.3}", score, cur_score, temp);
            cur_score = score;
            cur_adr = sample;
            cur_result = result;
            fail_counter = 0;
            fail_counter_target = initial_fail_counter_target;
        } else {
            println!("Do nothing.");
            fail_counter += 1;
            if fail_counter == fail_counter_target {
                let prev_score = cur_score;
                cur_result = eval(&cur_adr);
                cur_score = cur_result.efficiency_confidence() as f32 - get_memorized_penalty(&cur_result, it_ratio, baseline_memorized);
                fail_counter = 0;
                fail_counter_target = 2 * fail_counter_target;
                println!("Rerolling... {:.3} -> {:.3}, {:.3}", prev_score, cur_result.efficiency_confidence(), cur_score);
            }
        }
    }
    best_result = cur_result;
    best_adr = cur_adr;

    println!("Initial score: {}, Final score: {}, Baseline memorized: {}, Final memorized: {}", baseline_result.efficiency(), best_result.efficiency(), baseline_result.memorized(), best_result.memorized());
    // let verify = simulate(10000.0, deck_size, new_cards_per_day, days as f32, &predictor, &best_adr, &behavior_model, &mut rng);
    let verify = simulate(30000.0, deck_size, new_cards_per_day, days as f32, &predictor, &best_adr, &behavior_model, &mut rng);
    println!("Best score repeated: efficiency: {:.3} memorized: {:.3}", verify.efficiency(), verify.memorized());
    println!("Best adr: {:?}", best_adr);
    if baseline_result.efficiency() > verify.efficiency() {
        baseline_adr
    } else {
        best_adr
    }
}