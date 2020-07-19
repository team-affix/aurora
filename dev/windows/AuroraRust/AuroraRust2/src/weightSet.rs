#![allow(non_snake_case)]
use crate::model::*;
use crate::math::*;

#[derive(Clone)]
pub struct WeightSet {
    pub x: Carry,
    pub y: Carry,
    pub param: Carry
}

impl Model for WeightSet {
    fn fwd(&mut self, input: Carry) -> Carry {
        self.x = input.clone();
        let mut output = Carry::new(0.0);
        for i in 0..self.param.a.len() {
            output.a.push(
                Carry::new (
                    self.param.a[i].b * input.b
                )
            );
        }
        self.y = output.clone();
        output
    }
}

impl WeightSet {
    pub fn new(forParam: Carry) -> WeightSet {
        WeightSet { x: Carry::new(0.0), y: Carry::new(0.0), param: forParam }
    }
}
