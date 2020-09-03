#![allow(non_snake_case)]
use crate::math::*;

pub trait Model {
    fn fwd(&mut self, input: Carry) -> Carry;
    fn getX(&self) -> &Carry;
    fn getY(&self) -> &Carry;
}