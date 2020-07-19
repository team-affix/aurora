#![allow(non_snake_case)]

mod model;
mod layer;
mod math;
mod bias;
mod seq;
mod act;

#[derive(Clone)]
pub struct LeakyReLu {}
impl model::Model for LeakyReLu {
    fn fwd(&mut self, input: math::Carry) -> math::Carry {
        println!("hello from leaky relu");
        let mut y: f64 = input.b;
        if input.b < 0.0 {
            y = input.b * 0.05;
        }
        math::Carry::new(y)
    }
}

#[derive(Clone)]
pub struct Sigmoid {}
impl model::Model for Sigmoid {
    fn fwd(&mut self, input: math::Carry) -> math::Carry {
        println!("hello from Sigmoid");
        input
    }
}

pub fn makeLayer ( t: Vec<Box<dyn model::Model>>) -> layer::Layer {
    let mut theMods: Vec<Box<dyn model::Model>> = Vec::new();
    for xT in t {
        let currAct = act::Activation { base: xT, x: math::Carry::new(0.0), y: math::Carry::new(0.0) };
        theMods.push(Box::new(currAct));
    }

    layer::Layer { x: math::Carry::new(0.0), y: math::Carry::new(0.0), mods: theMods }
}

fn main() {
    let myLeaky = LeakyReLu{};
    
    let myLayer = makeLayer (
        vec![
            Box::new(myLeaky.clone()),
            Box::new(myLeaky.clone()),
            Box::new(myLeaky.clone()),
            Box::new(myLeaky.clone())
        ]
    );
    
    let mut mySeq = seq::Seq { x: math::Carry::new(0.0), y: math::Carry::new(0.0), mods: vec![Box::new(myLayer)] };
    let input = math::Carry {
        a: vec![
            math::Carry::new(1.3421),
            math::Carry::new(12.1234),
            math::Carry::new(423.22),
            math::Carry::new(0.23)
        ],
        b: 0.0
    };
    let output = mySeq.fwd(input);
    println!("{:?}", output);
}
