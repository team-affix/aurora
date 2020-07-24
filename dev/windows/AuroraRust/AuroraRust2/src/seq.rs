#![allow(non_snake_case)]
use crate::model::*;
use crate::math::*;

pub struct Seq {
    pub x: Box<Carry>,
    pub y: Box<Carry>,
    pub models: Vec<Box<dyn Model>>
}

impl Model for Seq {
    fn fwd(&mut self, input: Carry) -> Carry {
        runSeq(self, input)
    }
    fn getX(&mut self) -> Carry {
        self.x.
    }
    fn getY(&mut self) -> Carry {
        self.y
    }
}

impl Seq {
    pub fn new(forModels: Vec<Box<dyn Model>>) -> Seq {
        Seq {
            x: Carry::new(0.0),
            y: Carry::new(0.0),
            models: forModels
        }
    }
    
    pub fn fwd(&mut self, input: Carry) -> Carry {
        runSeq(self, input)
    }
}

fn runSeq(personal: &mut Seq, input: Carry) -> Carry {
    for i in 1..personal.models.len() {
        personal.models[i].fwd (
            personal.models[i-1].getY()
        );
    }
    personal.x = input;
    personal.y = personal.models[personal.models.len()-1].getY();
    personal.models[personal.models.len()-1].getY()
}
