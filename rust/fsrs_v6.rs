type FSRSv6Params = [f32; 21];
pub struct FSRSv6 {
    w: FSRSv6Params,
}
pub struct FSRSv6State {
    pub s: f32,
    pub d: f32,
}
impl FSRSv6State {
    pub fn to_tuple(&self) -> (f32, f32) {
        (self.s, self.d)
    }
}

impl FSRSv6 {
    const MIN_STABILITY: f32 = 0.1;
    const MAX_STABILITY: f32 = 365.0 * 25.0;

    pub fn new(params: FSRSv6Params) -> Self {
        Self { w: params }
    }
    fn init_d(&self, rating: i32) -> f32 {
        (self.w[4] - (self.w[5] * (rating - 1) as f32).exp() + 1.0).clamp(1.0, 10.0)
    }
    pub fn first_review(&self, rating: i32) -> FSRSv6State {
        let s = self.w[rating as usize - 1];
        let d = self.init_d(rating);
        FSRSv6State { s: s, d: d }
    }
    fn forgetting_curve(&self, state: &FSRSv6State, elapsed: f32) -> f32 {
        let decay = -self.w[20];
        let factor = 0.9_f32.powf(1.0 / decay) - 1.0;
        (1.0 + factor * elapsed / state.s).powf(decay)
    }
    fn stability_short_term(&self, s: f32, rating: i32) -> f32 {
        let sinc = (self.w[17] * (rating as f32 - 3.0 + self.w[18])).exp()
            * s.powf(-self.w[19]);
        if rating >= 3 { // TODO 2?
            s * f32::max(1.0, sinc)
        } else {
            s * sinc
        }
    }
    fn stability_after_success(&self, s: f32, r: f32, d: f32, rating: i32) -> f32 {
        let hard_penalty = if rating == 2 { self.w[15] } else { 1.0 };
        let easy_bonus = if rating == 4 { self.w[16] } else { 1.0 };
        let sinc = 
            self.w[8].exp()
            * (11.0 - d)
            * (s.powf(-self.w[9]))
            * (((1.0 - r) * self.w[10]).exp() - 1.0);
        s * (1.0 + sinc * hard_penalty * easy_bonus)
    }
    fn stability_after_failure(&self, s: f32, r: f32, d: f32) -> f32 {
        let new_s = 
            self.w[11]
            * (d.powf(-self.w[12]))
            * ((s + 1.0).powf(self.w[13]) - 1.0)
            * ((1.0 - r) * self.w[14]).exp();
        let new_min = s / (self.w[17] * self.w[18]).exp();
        return f32::min(new_s, new_min)
    }
    fn linear_damping(&self, delta_d: f32, old_d: f32) -> f32 {
        return delta_d * (10.0 - old_d) / 9.0
    }
    fn mean_reversion(&self, weight: f32, init: f32, current: f32) -> f32 {
        return weight * init + (1.0 - weight) * current
    }
    fn next_d(&self, d: f32, rating: i32) -> f32 {
        let delta_d = -self.w[6] * (rating - 3) as f32;
        let new_d = d + self.linear_damping(delta_d, d);
        let new_d = self.mean_reversion(self.w[7], self.init_d(4), new_d);
        new_d.clamp(1.0, 10.0)
    }
    pub fn transition(&self, state: FSRSv6State, rating: i32, elapsed: f32) -> FSRSv6State {
        let r = self.forgetting_curve(&state, elapsed);
        let s = if elapsed < 1.0 {
            self.stability_short_term(state.s, rating)
        } else if rating > 1 {
            self.stability_after_success(state.s, r, state.d, rating)
        } else {
            self.stability_after_failure(state.s, r, state.d)
        };
        let s = s.clamp(Self::MIN_STABILITY, Self::MAX_STABILITY);
        let d = self.next_d(state.d, rating);
        FSRSv6State { s: s, d: d }
    }
    pub fn get_interval(&self, state: &FSRSv6State, dr: f32) -> f32 {
        let decay = -self.w[20];
        let factor = 0.9_f32.powf(1.0 / decay) - 1.0;
        f32::max(1.0, state.s / factor * (dr.powf(1.0 / decay) - 1.0))
    }
}