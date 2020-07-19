#![allow(non_snake_case)]
use crate::weightSet::*;
use crate::model::*;
use crate::math::*;

#[derive(Clone)]
pub struct WeightJunction {
    pub x: Carry,
    pub y: Carry,
    pub weightSets: Vec<WeightSet>
}

impl Model for WeightJunction {
    fn fwd(&mut self, input: Carry) -> Carry {
        self.x = input.clone();

        let mut output = self.weightSets[0].fwd(input.a[0].clone());
        for i in 1..self.weightSets.len() {
            addTwoVec(
                self.weightSets[i].fwd (input.a[i].clone()),
                output.clone(),
                &mut output
            );
        }
        self.y = output.clone();
        output
    }
}

impl WeightJunction {
    pub fn new(forWeightSet: Vec<WeightSet>) -> WeightJunction {
        WeightJunction { x: Carry::new(0.0), y: Carry::new(0.0), weightSets: forWeightSet }
    }
}
