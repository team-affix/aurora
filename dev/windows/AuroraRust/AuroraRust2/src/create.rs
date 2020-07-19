#![allow(non_snake_case)]
use crate::weightJunction::*;
use crate::weightSet::*;
use crate::layer::*;
use crate::model::*;
use crate::math::*;
use crate::bias::*;
use crate::seq::*;
use crate::act::*;

use rand::Rng;

pub fn newLayer<T: Act + Clone + 'static> (nueronAmount: i32, layerActivation: T) -> Layer {
    // Basically neurons
    let mut userSeqs: Vec<Box<dyn Model>> = Vec::new();
    let mut rng = rand::thread_rng();
    for _i in 0..nueronAmount {
        let currBias = Bias::new(
            rng.gen_range(-1.0, 1.0)
        );
        let currAct = Activation::new(layerActivation.clone());
        let currSeq = Seq::new(vec![Box::new(currBias), Box::new(currAct)]);
        userSeqs.push(Box::new(currSeq));
    }
    Layer::new(userSeqs)
}

pub fn newWeightJunction(startLayer: i32, endLayer: i32) -> WeightJunction{
    let mut userSets: Vec<WeightSet> = Vec::new();
    let mut rng = rand::thread_rng();

    for _i in 0..startLayer {
        let mut currParams: Carry = Carry::new(0.0);
        for _j in 0..endLayer {
            currParams.a.push(Carry::new(
                rng.gen_range(-1.0, 1.0)
            ));
        }
        let currSet = WeightSet::new(currParams);
        userSets.push(currSet);
    }
    
    WeightJunction::new(userSets)
}
