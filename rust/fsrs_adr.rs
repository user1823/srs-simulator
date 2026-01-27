use std::f32::consts::TAU;

use arrayvec::ArrayVec;
use rand::Rng;

#[derive(Copy, Clone, Debug)]
struct DecisionPlane {
    mu_s: f32,
    mu_d: f32,
    dir_s: f32,
    dir_d: f32,
}
impl DecisionPlane {
    fn new(mu_s: f32, mu_d: f32, theta: f32) -> Self {
        let (sin_t, cos_t) = theta.sin_cos();
        Self {
            mu_s,
            mu_d,
            dir_s: cos_t,
            dir_d: sin_t,
        }
    }
    fn empty() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    #[inline]
    fn prop(&self, s: f32, d: f32) -> bool {
        let dot_prod = (s - self.mu_s) * self.dir_s + (d - self.mu_d) * self.dir_d;
        return dot_prod > 0.0
    }
    fn gen<T: Rng>(rng: &mut T) -> Self {
        let mu_s = rng.random_range(-7.0..10.0);
        let mu_d = rng.random_range(1.0..10.0);
        let theta = rng.random_range(0.0..TAU);
        Self::new(mu_s, mu_d, theta)
    }
    fn mutate<T: Rng>(&mut self, rng: &mut T) {
        match rng.random_range(0..3) {
            0 => self.mu_s = (self.mu_s + rng.random_range(-1.0..1.0)).clamp(-12.0, 15.0),
            1 => self.mu_d = (self.mu_d + rng.random_range(-1.0..1.0)).clamp(-5.0, 15.0),
            2 => {
                let angle: f32 = rng.random_range(-0.5..0.5);
                let (s, c) = angle.sin_cos();

                let new_s = self.dir_s * c - self.dir_d * s;
                let new_d = self.dir_s * s + self.dir_d * c;

                self.dir_s = new_s;
                self.dir_d = new_d;
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Debug)]
struct RegionBonus {
    bonus: f32,
    decision_planes: ArrayVec<DecisionPlane, 3>
}

impl RegionBonus {
    #[inline]
    fn eval(&self, s: f32, d: f32) -> f32 {
        if self.decision_planes.iter().all(|x| x.prop(s, d)) {
            self.bonus
        } else {
            0.0
        }
    }
    fn gen<T: Rng>(rng: &mut T) -> Self {
        let bonus = rng.random_range(-0.2..0.2);
        let n_planes = rng.random_range(1..=3);
        let mut decision_planes = ArrayVec::new();
        for _ in 0..n_planes {
            decision_planes.push(DecisionPlane::gen(rng));
        }
        Self {
            bonus,
            decision_planes,
        }
    }
    fn mutate<T: Rng>(&mut self, rng: &mut T) {
        let r = rng.random_range(1..100);

        if r < 30 {
            self.bonus = (self.bonus + rng.random_range(-0.2..0.2)).clamp(-3.0, 3.0);
        } else if r < 70 {
            let i = rng.random_range(0..self.decision_planes.len());
            self.decision_planes[i].mutate(rng);
        } else {
            if rng.random_bool(0.5) {
                if self.decision_planes.len() < self.decision_planes.capacity() {
                    self.decision_planes.push(DecisionPlane::gen(rng));
                }
            } else {
                if self.decision_planes.len() > 1 {
                    let i = rng.random_range(0..self.decision_planes.len());
                    self.decision_planes.swap_remove(i);
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Gaussian {
    pub amplitude: f32,
    pub mu_s: f32,
    pub mu_d: f32,
    pub sigma_s: f32,
    pub sigma_d: f32,
    pub theta: f32,
    inv_s: f32,
    inv_d: f32,
    sin_t: f32,
    cos_t: f32,
}

impl Gaussian {
    fn new(amplitude: f32, mu_s: f32, mu_d: f32, sigma_s: f32, sigma_d: f32, theta: f32) -> Self {
        let (sin_t, cos_t) = theta.sin_cos();
        Self {
            amplitude,
            mu_s,
            mu_d,
            sigma_s,
            sigma_d,
            theta,
            inv_s: 1.0 / (sigma_s * sigma_s),
            inv_d: 1.0 / (sigma_d * sigma_d),
            sin_t,
            cos_t,
        }
    }
    fn empty() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0, 1.0, 0.0)
    }
    #[inline]
    fn eval(&self, s: f32, d: f32) -> f32 {
        let ds: f32 = s - self.mu_s;
        let dd = d - self.mu_d;
        let x =  self.cos_t * ds + self.sin_t * dd;
        let y = -self.sin_t * ds + self.cos_t * dd;
        let q = x * x * self.inv_s + y * y * self.inv_d;
        self.amplitude * (-q).exp()
    }
}

#[derive(Clone, Debug)]
pub struct FSRSADR {
    flat: f32,
    gaussians: ArrayVec<Gaussian, 8>,
    decision_bonuses: ArrayVec<RegionBonus, 8>,
}
impl FSRSADR {
    pub fn new(dr: f32) -> Self {
        Self::fixed_dr(dr)
    }
    pub fn fixed_dr(dr: f32) -> Self {
        Self { flat: inverse_sigmoid(dr), gaussians: ArrayVec::new(), decision_bonuses: ArrayVec::new() }
    }
    pub fn get_dr(&self, s: f32, d: f32) -> f32 {
        let s = s.ln();
        let logit = 
            self.flat 
            + self.gaussians.iter().map(|g| g.eval(s, d)).sum::<f32>()
            + self.decision_bonuses.iter().map(|g| g.eval(s, d)).sum::<f32>();
        sigmoid(logit).clamp(0.0, 0.995)
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    let x = x.clamp(-10.0, 10.0);
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn inverse_sigmoid(x: f32) -> f32 {
    let x = x.clamp(1e-6, 1.0 - 1e-6);
    (x / (1.0 - x)).ln()
}

pub struct FSRSADRGenerator {
    init: bool
}

impl FSRSADRGenerator {
    pub fn new() -> Self {
        Self { init: false }
    }
    pub fn suggest<T: Rng>(&mut self, adr_model: &FSRSADR, rng: &mut T) -> FSRSADR {
        let mut adr_clone = adr_model.clone();
        let choice: u32 = rng.random_range(0..100);
        if adr_clone.decision_bonuses.is_empty() || (choice < 20 && adr_model.decision_bonuses.len() < adr_model.decision_bonuses.capacity()) {
            Self::add_decision_bonus(&mut adr_clone, rng);
        } else if choice < 400 {
            Self::adjust_decision_bonus(&mut adr_clone, rng);
        } else if choice < 50 {
            Self::adjust_flat(&mut adr_clone, rng);
        } else if (choice < 60 || adr_model.gaussians.is_empty()) && adr_model.gaussians.len() < adr_model.gaussians.capacity() {
            Self::add_gaussian(&mut adr_clone, rng);
        } else if choice <= 70 && !adr_model.gaussians.is_empty() {
            Self::delete_gaussian(&mut adr_clone, rng);
        } else {
            Self::adjust_gaussian(&mut adr_clone, rng);
        }
        adr_clone
    }
    fn adjust_flat<T: Rng>(adr_model: &mut FSRSADR, rng: &mut T) -> () {
        let offset: f32 = rng.random_range(-0.1..0.1);
        adr_model.flat += offset;
        adr_model.flat = f32::min(adr_model.flat, inverse_sigmoid(0.99));
    }
    fn add_gaussian<T: Rng>(adr_model: &mut FSRSADR, rng: &mut T) -> () {
        adr_model.gaussians.push(Self::gen_gaussian(rng));
    }
    fn delete_gaussian<T: Rng>(adr_model: &mut FSRSADR, rng: &mut T) -> () {
        let idx = rng.random_range(0..adr_model.gaussians.len());
        adr_model.gaussians.swap_remove(idx);
    }
    fn gen_gaussian<T: Rng>(rng: &mut T) -> Gaussian {
        Gaussian::new(
            Self::gen_amplitude(rng),
            Self::gen_mu_s(rng),
            Self::gen_mu_d(rng),
            Self::gen_sigma_s(rng),
            Self::gen_sigma_d(rng),
            Self::gen_theta(rng),
        )
    }
    fn adjust_gaussian<T: Rng>(adr_model: &mut FSRSADR, rng: &mut T) {
        let rand_index = rng.random_range(0..adr_model.gaussians.len());
        let gaussian = unsafe { adr_model.gaussians.get_mut(rand_index).unwrap_unchecked() };
        // Copy current values
        let mut amplitude = gaussian.amplitude;
        let mut mu_s = gaussian.mu_s;
        let mut mu_d = gaussian.mu_d;
        let mut sigma_s = gaussian.sigma_s;
        let mut sigma_d = gaussian.sigma_d;
        let mut theta = gaussian.theta;

        match rng.random_range(0..6) {
            0 => amplitude = Self::mod_amplitude(rng, amplitude),
            1 => mu_s      = Self::mod_mu_s(rng, mu_s),
            2 => mu_d      = Self::mod_mu_d(rng, mu_d),
            3 => sigma_s   = Self::mod_sigma_s(rng, sigma_s),
            4 => sigma_d   = Self::mod_sigma_d(rng, sigma_d),
            5 => theta     = Self::mod_theta(rng, theta),
            _ => unreachable!(),
        }

        *gaussian = Gaussian::new(amplitude, mu_s, mu_d, sigma_s, sigma_d, theta);
    }
    fn gen_amplitude<T: Rng>(rng: &mut T) -> f32 {
        rng.random_range(-0.2..0.2)
    }
    fn mod_amplitude<T: Rng>(rng: &mut T, x: f32) -> f32 {
        (x + rng.random_range(-0.2..0.2)).clamp(-2.0, 2.0)
    }
    fn gen_mu_s<T: Rng>(rng: &mut T) -> f32 {
        rng.random_range(-7.0..10.0)
    }
    fn mod_mu_s<T: Rng>(rng: &mut T, x: f32) -> f32 {
        (x + rng.random_range(-1.0..1.0)).clamp(-8.0, 11.0)
    }
    fn gen_mu_d<T: Rng>(rng: &mut T) -> f32 {
        rng.random_range(0.0..11.0)
    }
    fn mod_mu_d<T: Rng>(rng: &mut T, x: f32) -> f32 {
        (x + rng.random_range(-1.0..1.0)).clamp(-1.0, 12.0)
    }
    fn gen_sigma_s<T: Rng>(rng: &mut T) -> f32 {
        rng.random_range(-0.5f32..3.0).exp()
    }
    fn mod_sigma_s<T: Rng>(rng: &mut T, x: f32) -> f32 {
        (x.ln() + rng.random_range(-0.2..0.2)).exp()
    }
    fn gen_sigma_d<T: Rng>(rng: &mut T) -> f32 {
        rng.random_range(-0.5f32..3.0).exp()
    }
    fn mod_sigma_d<T: Rng>(rng: &mut T, x: f32) -> f32 {
        (x.ln() + rng.random_range(-0.2..0.2)).exp()
    }
    fn gen_theta<T: Rng>(rng: &mut T) -> f32 {
        rng.random_range(0.0..TAU)
    }
    fn mod_theta<T: Rng>(rng: &mut T, x: f32) -> f32 {
        (x + rng.random_range(-0.5..0.5)).rem_euclid(TAU)
    }
    fn add_decision_bonus<T: Rng>(adr_model: &mut FSRSADR, rng: &mut T) -> () {
        adr_model.decision_bonuses.push(RegionBonus::gen(rng));
    }
    fn adjust_decision_bonus<T: Rng>(adr_model: &mut FSRSADR, rng: &mut T) {
        let rand_index = rng.random_range(0..adr_model.decision_bonuses.len());
        let decision_bonus = unsafe { adr_model.decision_bonuses.get_unchecked_mut(rand_index) };
        decision_bonus.mutate(rng);
    }
    pub fn record_score(&self, score: f64) -> () {

    }
}