#![allow(non_snake_case)]
use crate::model::*;
use crate::math::*;

pub struct Seq {
    pub x: Carry,
    pub y: Carry,
    pub mods: Vec<Box<dyn Model>>
}


impl Model for Seq {
    fn fwd(&mut self, input: Carry) -> Carry{
        let mut currOutput = self.mods[0].fwd(input.clone());
        for x in &mut self.mods {
            currOutput = x.fwd(currOutput);
        }
        self.x = input;
        self.y = currOutput.clone();
        currOutput   
    }
}


impl Seq {
    pub fn fwd(&mut self, input: Carry) -> Carry {
        let mut currOutput = self.mods[0].fwd(input.clone());
        for x in &mut self.mods {
            currOutput = x.fwd(currOutput);
        }
        self.x = input;
        self.y = currOutput.clone();
        currOutput
    }
}
