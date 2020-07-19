#![allow(non_snake_case)]
use crate::model::*;
use crate::math::*;

pub struct Seq {
    pub x: Carry,
    pub y: Carry,
    pub models: Vec<Box<dyn Model>>
}

impl Model for Seq {
    fn fwd(&mut self, input: Carry) -> Carry {
        runSeq(self, input)
    }
}

impl Seq {
    pub fn new(forModels: Vec<Box<dyn Model>>) -> Seq {
        Seq { x: Carry::new(0.0), y: Carry::new(0.0), models: forModels }
    }
    
    pub fn fwd(&mut self, input: Carry) -> Carry {
        runSeq(self, input)
    }
}

fn runSeq(personal: &mut Seq, input: Carry) -> Carry {
    personal.x = input.clone();    
    let mut output = input;
    for xModels in &mut personal.models {
        output = xModels.fwd(output);
    }
    personal.y = output.clone();
    output
}
