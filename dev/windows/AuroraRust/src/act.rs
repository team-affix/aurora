#[allow(non_snake_case)]
use crate::model::*;
use crate::math::*;

// pub trait Act{ fn act(&mut self); }

// #[derive(Clone)]
// pub struct Activation<T: Model> {
//     pub base: T,
//     pub x: f64,
//     pub y: f64
// }

// impl<T: Model> Model for Activation<T> {
//     fn fwd(&mut self) {
//         self.base.fwd();
//     }
// }


pub struct Activation {
    pub base: Box<dyn Model>,
    pub x: Carry,
    pub y: Carry
}

impl Model for Activation {
    fn fwd(&mut self, input: Carry) -> Carry {
        self.x = input.clone();
        self.y = self.base.fwd( input );
        self.y.clone()
    }
}

impl Activation {
    
}
