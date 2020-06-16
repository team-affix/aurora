#pragma once
#include "superHeader.h"
#include "optimization.h"
#include "maths.h"
using namespace maths;
using namespace optimization;

namespace modeling {
	class model;
	class model_bpg;
	class bias;
	class bias_bpg;
	class activate;
	class activate_bpg;
	class weight;
	class weight_bpg;
	class weightSet;
	class weightSet_bpg;
	class weightJunction;
	class weightJunction_bpg;
	class sequential;
	class sequential_bpg;
	class layer;
	class layer_bpg;
	void initialize_model(model* m, vector<parameter**>* parameters);
	void compile_model(model* m);
#pragma region model
	static void model_forward(shared_ptr<carryType> x, shared_ptr<carryType> y);
	static void model_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad);
	static void model_foreach(function<void(model*)> func, model* m);
#pragma endregion
#pragma region bias
	static void bias_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, parameter* param);
	static void bias_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad, parameter* param);
	static void bias_deconstruct(parameter** param);
#pragma endregion
#pragma region activate
	static void activate_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, activation* act);
	static void activate_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> y, shared_ptr<carryType> xGrad, activation* act);
	static void activate_deconstruct(activation* act);
#pragma endregion
#pragma region weight
	static void weight_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, parameter* param);
	static void weight_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad, shared_ptr<carryType> x, parameter* param);
	static void weight_deconstruct(parameter** param);
#pragma endregion
#pragma region weightSet
	static void weightSet_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, vector<model*>* weights);
	static void weightSet_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad, vector<model*>* weights);
	static void weightSet_foreach(function<void(model*)> func, model* m, vector<model*>* models);
	static void weightSet_deconstruct(vector<model*>* models);
#pragma endregion
#pragma region weightJunction
	static void weightJunction_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, int* b, vector<model*>* models);
	static void weightJunction_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad, vector<model*>* models);
	static void weightJunction_foreach(function<void(model*)> func, model* m, vector<model*>* models);
	static void weightJunction_deconstruct(vector<model*>* models);
#pragma endregion
#pragma region sequential
	static void sequential_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, vector<model*>* models);
	static void sequential_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad, vector<model*>* models);
	static void sequential_foreach(function<void(model*)> func, model* m, vector<model*>* models);
	static void sequential_deconstruct(vector<model*>* models);
#pragma endregion
#pragma region layer
	static void layer_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, vector<model*>* models);
	static void layer_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad, vector<model*>* models);
	static void layer_foreach(function<void(model*)> func, model* m, vector<model*>* models);
	static void layer_deconstruct(vector<model*>* models);
#pragma endregion
	class model {
	public:
		virtual ~model();
		model();
		virtual void fwd();
		virtual void foreach(function<void(model*)> func);
		virtual model* clone();
		shared_ptr<carryType> x;
		shared_ptr<carryType> y;
	};
	class model_bpg : public model {
	public:
		model_bpg();
		virtual void bwd();
		virtual model* clone();
		shared_ptr<carryType> xGrad;
		shared_ptr<carryType> yGrad;
	};
	class bias : public model {
	public:
		virtual ~bias();
		bias();
		virtual void fwd();
		virtual model* clone();
		parameter** param;
	};
	class bias_bpg : public model_bpg {
	public:
		virtual ~bias_bpg();
		bias_bpg();
		virtual void fwd();
		virtual void bwd();
		virtual model* clone();
		parameter** param;
	};
	class activate : public model {
	public:
		virtual ~activate();
		activate(activation* act);
		virtual void fwd();
		virtual model* clone();
		activation* act;
	};
	class activate_bpg : public model_bpg {
	public:
		virtual ~activate_bpg();
		activate_bpg(activation* act);
		virtual void fwd();
		virtual void bwd();
		virtual model* clone();
		activation* act;
	};
	class weight : public model {
	public:
		virtual ~weight();
		weight();
		virtual void fwd();
		virtual model* clone();
		parameter** param;
	};
	class weight_bpg : public model_bpg {
	public:
		virtual ~weight_bpg();
		weight_bpg();
		virtual void fwd();
		virtual void bwd();
		virtual model* clone();
		parameter** param;
	};
	class weightSet : public model, public vector<model*> {
	public:
		virtual ~weightSet();
		weightSet(int a);
		virtual void fwd();
		virtual void foreach(function<void(model*)> func);
		virtual model* clone();
	};
	class weightSet_bpg : public model_bpg, public vector<model*> {
	public:
		virtual ~weightSet_bpg();
		weightSet_bpg(int a);
		virtual void fwd();
		virtual void bwd();
		virtual void foreach(function<void(model*)> func);
		virtual model* clone();
	};
	class weightJunction : public model, public vector<model*> {
	public:
		virtual ~weightJunction();
		weightJunction(int a, int b);
		virtual void fwd();
		virtual void foreach(function<void(model*)> func);
		virtual model* clone();
		int a;
		int b;
	};
	class weightJunction_bpg : public model_bpg, public vector<model*> {
	public:
		virtual ~weightJunction_bpg();
		weightJunction_bpg(int a, int b);
		virtual void fwd();
		virtual void bwd();
		virtual void foreach(function<void(model*)> func);
		virtual model* clone();
		int a;
		int b;
	};
	class sequential : public model, public vector<model*> {
	public:
		virtual ~sequential();
		sequential();
		virtual void fwd();
		virtual void foreach(function<void(model*)> func);
		virtual model* clone();
	};
	class sequential_bpg : public model_bpg, public vector<model*> {
	public:
		virtual ~sequential_bpg();
		sequential_bpg();
		virtual void fwd();
		virtual void bwd();
		virtual void foreach(function<void(model*)> func);
		virtual model* clone();
	};
	class layer : public model, public vector<model*> {
	public:
		virtual ~layer();
		layer();
		layer(int a, model* model_default);
		virtual void fwd();
		virtual void foreach(function<void(model*)> func);
		virtual model* clone();
	};
	class layer_bpg : public model_bpg, public vector<model*> {
	public:
		virtual ~layer_bpg();
		layer_bpg();
		layer_bpg(int a, model* model_default);
		virtual void fwd();
		virtual void bwd();
		virtual void foreach(function<void(model*)> func);
		virtual model* clone();
	};
}