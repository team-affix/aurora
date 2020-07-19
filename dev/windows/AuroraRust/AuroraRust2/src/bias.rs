#![allow(non_snake_case)]
use crate::model::*;
use crate::math::*;

#[derive(Clone)]
pub struct Bias {
    pub x: f64,
    pub y: f64,
    pub param: f64
}

impl Model for Bias {
    fn fwd(&mut self, input: Carry) -> Carry {
        self.x = input.b.clone();
        let output = self.param + input.b;

        self.y = output;
        // println!("{}", self.x);
        // println!("{}", output);
        Carry::new(output)
    }
}

impl Bias {
    pub fn new(forParam: f64) -> Bias {
        Bias { x: 0.0, y: 0.0, param: forParam }
    }
}
