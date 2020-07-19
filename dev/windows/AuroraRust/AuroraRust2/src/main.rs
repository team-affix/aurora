#![allow(non_snake_case)]

mod weightJunction;
mod weightSet;
mod create;
mod layer;
mod model;
mod math;
mod bias;
mod seq;
mod act;

#[derive(Clone)]
pub struct LeakyReLu {}
impl act::Act for LeakyReLu {
    fn act(input: f64) -> f64 {
        let mut output = input;
        if input < 0.0 {
            output = input * 0.05;
        }
        println!("{:?}", output);
        output
    } 
}

pub struct TNN {
    net: Box<dyn model::Model>
}
impl TNN {
    pub fn new(forNet: Box<dyn model::Model>) -> TNN {
        TNN { net: forNet }
    }
    pub fn fwd(&mut self, input: math::Carry) -> math::Carry {
        self.net.fwd(input)
    }
}

fn main() {
    let myLeakyReLu = LeakyReLu {};

    let layer1 = create::newLayer(2, myLeakyReLu.clone());
    let weightJunction1 = create::newWeightJunction(2, 3);
    let layer2 = create::newLayer(3, myLeakyReLu.clone());

    let modelsVec: Vec<Box<dyn model::Model>> = vec! [
        Box::new(layer1),
        Box::new(weightJunction1),
        Box::new(layer2)
    ];
    let mut myTNN = seq::Seq::new(modelsVec);
    
    let input = math::Carry {
        a: vec![math::Carry::new(4.0), math::Carry::new(10.0)],
        b: 0.0
    };

    println!("{:?}", input);
    let output = myTNN.fwd(input);
    println!("{:?}", output);
}
