#![allow(non_snake_case)]

#[derive(Debug, Clone)]
pub struct Carry { pub a: Vec<Carry>, pub b: f64 }

impl Carry {
    pub fn new(xB: f64) -> Carry {
        Carry { a: Vec::new(), b: xB }
    }
}
