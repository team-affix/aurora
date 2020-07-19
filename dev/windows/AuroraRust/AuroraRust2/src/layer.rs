#![allow(non_snake_case)]
use crate::math::*;
use crate::model::*;

pub struct Layer {
    pub x: Carry,
    pub y: Carry,
    pub models: Vec<Box<dyn Model>>
}

impl Model for Layer {
    fn fwd(&mut self, input: Carry) -> Carry {
        self.x = input.clone();
        let mut output = Carry::new(0.0);
        for i in 0..self.models.len() {
            output.a.push(self.models[i].fwd(input.a[i].clone()));
            // println!("{:?}", self.models[i].fwd(input.a[i].clone()));
        }
        self.y = output.clone();
        output
    }
}

impl Layer {
    pub fn new(forModels: Vec<Box<dyn Model>>) -> Layer {
        Layer { x: Carry::new(0.0), y: Carry::new(0.0), models: forModels }
    }
}
