#![allow(non_snake_case)]
use crate::model::*;
use crate::math::*;

pub struct Layer {
    pub x: Carry,
    pub y: Carry,
    pub mods: Vec<Box<dyn Model>>
}

impl Model for Layer {
    fn fwd(&mut self, input: Carry) -> Carry {
        self.x = input.clone();
        let mut output = Carry::new(0.0);
        for i in 0..self.mods.len() {
            output.a.push(
                self.mods[i].fwd( input.a[0].clone() )
            );
        }

        self.y = output.clone();
        output
    }
}

impl Layer {
    
}
