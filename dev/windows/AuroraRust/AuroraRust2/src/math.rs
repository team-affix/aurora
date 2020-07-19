#![allow(non_snake_case)]

#[derive(Debug, Clone)]
pub struct Carry {
    pub a: Vec<Carry>,
    pub b: f64
}

impl Carry {
    pub fn new(forB: f64) -> Carry {
        Carry { a: Vec::new(), b: forB }
    }
}

pub fn addTwoVec(first: Carry, second: Carry, output: &mut Carry) {
    let mut result: Carry = Carry::new(0.0);
    for i in 0..first.a.len() {
        result.a.push (
            Carry::new (
                first.a[i].b + second.a[i].b
            )
        );
    }
    *output = result;
}
