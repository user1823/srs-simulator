use std::f32::consts::TAU;

use rand::Rng;

use crate::fsrs_v6::FSRSv6State;

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

    #[inline]
    fn eval(&self, s: f32, d: f32) -> f32 {
        let ds = s - self.mu_s;
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
    gaussians: Vec<Gaussian>,
}
impl FSRSADR {
    pub fn new(dr: f32) -> Self {
        Self { flat: inverse_sigmoid(dr), gaussians: Vec::new() }
    }
    pub fn fixed_dr(dr: f32) -> Self {
        Self { flat: inverse_sigmoid(dr), gaussians: Vec::new() }
    }
    pub fn get_dr(&self, s: f32, d: f32) -> f32 {
        let s = s.ln();
        let logit = 
            self.flat 
            + self.gaussians.iter().map(|g| g.eval(s, d)).sum::<f32>();
        sigmoid(logit).clamp(0.0, 0.99)
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

}

impl FSRSADRGenerator {
    pub fn suggest<T: Rng>(&self, adr_model: &FSRSADR, rng: &mut T) -> FSRSADR {
        let choice: u32 = rng.random_range(0..100);
        let mut adr_clone = adr_model.clone();
        if choice < 10 {
            Self::adjust_flat(&mut adr_clone, rng);
        } else if (choice < 20 || adr_model.gaussians.is_empty()) && adr_model.gaussians.len() < 64 {
            Self::add_gaussian(&mut adr_clone, rng);
        } else if choice <= 30 && !adr_model.gaussians.is_empty() {
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
        let gaussian = Gaussian::new(
            Self::gen_amplitude(rng),
            Self::gen_mu_s(rng),
            Self::gen_mu_d(rng),
            Self::gen_sigma_s(rng),
            Self::gen_sigma_d(rng),
            Self::gen_theta(rng),
        );
        adr_model.gaussians.push(gaussian);
    }
    fn delete_gaussian<T: Rng>(adr_model: &mut FSRSADR, rng: &mut T) -> () {
        let idx = rng.random_range(0..adr_model.gaussians.len());
        adr_model.gaussians.remove(idx);
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
            0 => amplitude = amplitude + Self::gen_amplitude(rng),
            1 => mu_s      = Self::gen_mu_s(rng),
            2 => mu_d      = Self::gen_mu_d(rng),
            3 => sigma_s   = Self::gen_sigma_s(rng),
            4 => sigma_d   = Self::gen_sigma_d(rng),
            5 => theta     = Self::gen_theta(rng),
            _ => unreachable!(),
        }

        // Construct the new Gaussian once
        *gaussian = Gaussian::new(amplitude, mu_s, mu_d, sigma_s, sigma_d, theta);
    }

    fn gen_amplitude<T: Rng>(rng: &mut T) -> f32 {
        rng.random_range(-0.2..0.2)
    }
    fn gen_mu_s<T: Rng>(rng: &mut T) -> f32 {
        rng.random_range(-7.0..10.0)
    }
    fn gen_mu_d<T: Rng>(rng: &mut T) -> f32 {
        rng.random_range(-0.0..11.0)
    }
    fn gen_sigma_s<T: Rng>(rng: &mut T) -> f32 {
        rng.random_range(-0.5..3.0f32).exp()
    }
    fn gen_sigma_d<T: Rng>(rng: &mut T) -> f32 {
        rng.random_range(-0.5..3.0f32).exp()
    }
    fn gen_theta<T: Rng>(rng: &mut T) -> f32 {
        rng.random_range(0.0..TAU)
    }
    pub fn record_score(&self, score: f64) -> () {

    }
}