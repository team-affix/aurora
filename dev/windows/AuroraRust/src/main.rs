#![allow(non_snake_case)]

#[derive(Copy, Clone)]
struct Bias {
    param: f64
}

#[derive(Clone)]
struct Neuron {
    perBias: Bias,
    x: f64,
    y: f64,
}

fn main() {
    let myB = Bias {param: 5.5};
    println!("Bias: {}", &myB.param);

    let myN = Neuron {perBias: myB, x: 4.5, y: 6.5};
    
    let something = myN.clone();
    println!("Neuron: {} {} {}", myN.perBias.param, myN.x, myN.y);
}