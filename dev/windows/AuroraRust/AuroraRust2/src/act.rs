#![allow(non_snake_case)]
use crate::model::*;
use crate::math::*;

pub trait Act { fn act(input: f64) -> f64; }

#[derive(Clone)]
pub struct Activation<T: Act> {
    pub base: T,
    pub x: f64,
    pub y: f64
}

impl<T: Act> Model  for Activation<T> {
    fn fwd(&mut self, input: Carry) -> Carry {
        self.x = input.b.clone();
        let output = <T>::act(input.b);
        self.y = output;
        Carry::new(output)
    }
}

impl<T: Act> Activation<T> {
    pub fn new(t: T) -> Activation<T> {
        Activation { base: t, x: 0.0, y: 0.0 }
    }
}
