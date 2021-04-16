#pragma once
#include "pch.h"
#include "data.h"
#include "models.h"

using namespace aurora::models;

namespace aurora {
	namespace beautify {
		typedef ptr<model> Model;
		typedef ptr<bias> Bias;
		typedef ptr<weight> Weight;
		typedef ptr<sigmoid> Sigmoid;
		typedef ptr<models::tanh> Tanh;
		typedef ptr<leaky_relu> Leaky_relu;
		typedef ptr<layer> Layer;
		typedef ptr<sequential> Sequential;
		typedef ptr<weight_set> Weight_set;
		typedef ptr<weight_junction> Weight_junction;
		typedef ptr<sync> Sync;
		typedef ptr<lstm_ts> Lstm_ts;
		typedef ptr<lstm> Lstm;
		typedef ptr<cnl> Cnl;
	}
}