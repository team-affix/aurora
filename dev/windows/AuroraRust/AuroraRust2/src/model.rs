#![allow(non_snake_case)]
use crate::math::*;

pub trait Model {
    fn fwd(&mut self, input: Carry) -> Carry;
    fn getX(&mut self) -> Carry;
    fn getY(&mut self) -> Carry;
}