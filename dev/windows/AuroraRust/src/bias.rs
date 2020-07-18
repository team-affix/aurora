#![allow(non_snake_case)]
use crate::model::*;

struct Bias<T: Model> {
    base: T,
    x: f64,
    y: f64
}
