#pragma once
#include "modeling.h"
using namespace modeling;
//static functions
#pragma region model
static void modeling::model_forward(shared_ptr<carryType> x, shared_ptr<carryType> y) {

}
static void modeling::model_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad) {

}
static void modeling::model_foreach(function<void(model*)> func, model* m) {
	func(m);
}
#pragma endregion
#pragma region bias
static void modeling::bias_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, parameter* param) {
	double x_double = x->value_double;
	y->value_double = x_double + param->state;
}
static void modeling::bias_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad, parameter* param) {
	double yGrad_double = yGrad->value_double;
	parameter_sgd* param_sgd = (parameter_sgd*)param;
	param_sgd->gradient += yGrad_double;
	xGrad->value_double = yGrad_double;
}
static void modeling::bias_deconstruct(parameter** param) {
	if (param != NULL) {
		if (*param != NULL) {
			delete* param;
		}
		delete param;
	}
}
#pragma endregion
#pragma region activate
static void modeling::activate_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, activation* act) {
	double x_double = x->value_double;
	y->value_double = act->eval(x_double);
}
static void modeling::activate_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> y, shared_ptr<carryType> xGrad, activation* act) {
	double yGrad_double = yGrad->value_double;
	double deriv = act->deriv(y->value_double);
	xGrad->value_double = yGrad_double * deriv;
}
static void modeling::activate_deconstruct(activation* act) {
	if (act == NULL) {
		delete act;
	}
}
#pragma endregion
#pragma region weight
static void modeling::weight_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, parameter* param) {
	double x_double = x->value_double;
	y->value_double = x_double * param->state;
}
static void modeling::weight_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad, shared_ptr<carryType> x, parameter* param) {
	double yGrad_double = yGrad->value_double;
	double x_double = x->value_double;
	parameter_sgd* param_sgd = (parameter_sgd*)param;
	param_sgd->gradient += yGrad_double * x_double;
	xGrad->value_double = yGrad_double * param_sgd->state;
}
static void modeling::weight_deconstruct(parameter** param) {
	if (param != NULL) {
		if (*param != NULL) {
			delete* param;
		}
		delete param;
	}
}
#pragma endregion
#pragma region weightSet
static void modeling::weightSet_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, vector<model*>* models) {
	double x_double = x->value_double;
	vector<shared_ptr<carryType>>* y_vector = &y->value_vector;
	for (int i = 0; i < models->size(); i++) {
		model* m = models->at(i);
		m->x->value_double = x_double;
		m->fwd();
		y_vector->at(i)->value_double = m->y->value_double;
	}
}
static void modeling::weightSet_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad, vector<model*>* models) {
	vector<shared_ptr<carryType>>* yGrad_vector = &yGrad->value_vector;
	xGrad->value_double = 0;
	for (int i = 0; i < models->size(); i++) {
		model_bpg* m = (model_bpg*)models->at(i);
		m->yGrad->value_double = yGrad_vector->at(i)->value_double;
		m->bwd();
		xGrad->value_double += m->xGrad->value_double;
	}
}
static void modeling::weightSet_foreach(function<void(model*)> func, model* m, vector<model*>* models) {
	func(m);
	for (int i = 0; i < models->size(); i++) {
		models->at(i)->foreach(func);
	}
}
static void modeling::weightSet_deconstruct(vector<model*>* models) {
	for (int remainder = models->size(); remainder > 0; --remainder) {
		delete models->at(0);
		models->erase(models->begin());
	}
}
#pragma endregion
#pragma region weightJunction
static void modeling::weightJunction_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, int* b, vector<model*>* models) {
	vector<shared_ptr<carryType>>* x_vector = &x->value_vector;
	zero(&y->value_vector);
	for (int i = 0; i < models->size(); i++) {
		model* m = models->at(i);
		m->x->value_double = x_vector->at(i)->value_double;
		m->fwd();
		add(&y->value_vector, &m->y->value_vector, &y->value_vector);
	}
}
static void modeling::weightJunction_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad, vector<model*>* models) {
	vector<shared_ptr<carryType>>* xGrad_vector = &xGrad->value_vector;
	for (int i = 0; i < models->size(); i++) {
		model_bpg* m = (model_bpg*)models->at(i);
		m->yGrad->value_vector = yGrad->value_vector;
		m->bwd();
		xGrad_vector->at(i)->value_double = m->xGrad->value_double;
	}
}
static void modeling::weightJunction_foreach(function<void(model*)> func, model* m, vector<model*>* models) {
	func(m);
	for (int i = 0; i < models->size(); i++) {
		models->at(i)->foreach(func);
	}
}
static void modeling::weightJunction_deconstruct(vector<model*>* models) {
	for (int i = 0; i < models->size(); i++) {
		delete models->at(i);
	}
}
#pragma endregion
#pragma region sequential
static void modeling::sequential_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, vector<model*>* models) {
	shared_ptr<carryType> CY = shared_ptr<carryType>(x);
	for (int i = 0; i < models->size(); i++) {
		model* m = models->at(i);
		m->x = CY;
		m->fwd();
		CY = m->y;
	}
	*y = *CY;
}
static void modeling::sequential_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad, vector<model*>* models) {
	shared_ptr<carryType> CG = shared_ptr<carryType>(yGrad);
	for (int i = models->size() - 1; i >= 0; --i) {
		model_bpg* m = (model_bpg*)models->at(i);
		m->yGrad = CG;
		m->bwd();
		CG = m->xGrad;
	}
	*xGrad = *CG;
}
static void modeling::sequential_foreach(function<void(model*)> func, model* m, vector<model*>* models) {
	func(m);
	for (int i = 0; i < models->size(); i++) {
		models->at(i)->foreach(func);
	}
}
static void modeling::sequential_deconstruct(vector<model*>* models) {
	for (int remainder = models->size(); remainder > 0; remainder--) {
		delete models->at(0);
		models->erase(models->begin());
	}
}
#pragma endregion
#pragma region layer
static void modeling::layer_forward(shared_ptr<carryType> x, shared_ptr<carryType> y, vector<model*>* models) {
	vector<shared_ptr<carryType>>* x_vector = &x->value_vector;
	vector<shared_ptr<carryType>>* y_vector = &y->value_vector;
	for (int i = 0; i < models->size(); i++) {
		model* m = models->at(i);
		m->x = x_vector->at(i);
		m->fwd();
		y_vector->at(i) = m->y;
	}
}
static void modeling::layer_backward(shared_ptr<carryType> yGrad, shared_ptr<carryType> xGrad, vector<model*>* models) {
	vector<shared_ptr<carryType>>* yGrad_vector = &yGrad->value_vector;
	vector<shared_ptr<carryType>>* xGrad_vector = &xGrad->value_vector;
	for (int i = 0; i < models->size(); i++) {
		model_bpg* m = (model_bpg*)models->at(i);
		m->yGrad = yGrad_vector->at(i);
		m->bwd();
		xGrad_vector->at(i) = m->xGrad;
	}
}
static void modeling::layer_foreach(function<void(model*)> func, model* m, vector<model*>* models) {
	func(m);
	for (int i = 0; i < models->size(); i++) {
		models->at(i)->foreach(func);
	}
}
static void modeling::layer_deconstruct(vector<model*>* models) {
	for (int remainder = models->size(); remainder > 0; remainder--) {
		delete models->at(0);
		models->erase(models->begin());
	}
}
#pragma endregion
#pragma region functions
void modeling::initialize_model(model* m, vector<parameter**>* parameters) {
	if (bias* b = dynamic_cast<bias*>(m)) {
		b->param = new parameter * ();
		parameters->push_back(b->param);
	}
	else if (bias_bpg* b = dynamic_cast<bias_bpg*>(m)) {
		b->param = new parameter * ();
		parameters->push_back(b->param);
	}
	else if (weight* w = dynamic_cast<weight*>(m)) {
		w->param = new parameter * ();
		parameters->push_back(w->param);
	}
	else if (weight_bpg* b = dynamic_cast<weight_bpg*>(m)) {
		weight_bpg* w = (weight_bpg*)m;
		w->param = new parameter * ();
		parameters->push_back(w->param);
	}
}
void modeling::compile_model(model* m) {
	if (sequential* s = dynamic_cast<sequential*>(m)) {
		shared_ptr<carryType> CY = s->x;
		for (int i = 0; i < s->size(); i++) {
			model* mPtr = s->at(i);
			mPtr->x = CY;
			CY = mPtr->y;
		}
		s->y = CY;
	}
	else if (sequential_bpg* s = dynamic_cast<sequential_bpg*>(m)) {
		shared_ptr<carryType> CY = s->x;
		shared_ptr<carryType> CG = s->xGrad;
		for (int i = 0; i < s->size(); i++) {
			model_bpg* mPtr = (model_bpg*)s->at(i);
			mPtr->x = CY;
			mPtr->xGrad = CG;
			CY = mPtr->y;
			CG = mPtr->yGrad;
		}
		s->y = CY;
		s->yGrad = CG;
	}
	else if (layer* l = dynamic_cast<layer*>(m)) {
		for (int i = 0; i < l->size(); i++) {
			model* mPtr = l->at(i);
			mPtr->x = l->x->value_vector.at(i);
			l->y->value_vector.at(i) = mPtr->y;
		}
	}
	else if (layer_bpg* l = dynamic_cast<layer_bpg*>(m)) {
		for (int i = 0; i < l->size(); i++) {
			model_bpg* mPtr = (model_bpg*)l->at(i);
			mPtr->x = l->x->value_vector.at(i);
			mPtr->xGrad = l->xGrad->value_vector.at(i);
			l->y->value_vector.at(i) = mPtr->y;
			l->yGrad->value_vector.at(i) = mPtr->yGrad;
		}
	}
}
#pragma endregion
//class functions
model::~model() {

}
model::model() {
	x = shared_ptr<carryType>(new carryType());
	y = shared_ptr<carryType>(new carryType());
}
void model::fwd() {
	model_forward(x, y);
}
void model::foreach(function<void(model*)> func) {
	model_foreach(func, this);
}
model* model::clone() {
	return new model(*this);
}
model_bpg::model_bpg() {
	x = shared_ptr<carryType>(new carryType());
	y = shared_ptr<carryType>(new carryType());
	xGrad = shared_ptr<carryType>(new carryType());
	yGrad = shared_ptr<carryType>(new carryType());
}
void model_bpg::bwd() {
	model_backward(yGrad, xGrad);
}
model* model_bpg::clone() {
	return new model_bpg(*this);
}
bias::~bias() {
	bias_deconstruct(param);
}
bias::bias() {
	x = shared_ptr<carryType>(new carryType());
	y = shared_ptr<carryType>(new carryType());
}
void bias::fwd() {
	bias_forward(x, y, *param);
}
model* bias::clone() {
	bias* result = new bias(*this);
	return result;
}
bias_bpg::~bias_bpg() {
	bias_deconstruct(param);
}
bias_bpg::bias_bpg() {
	x = shared_ptr<carryType>(new carryType());
	y = shared_ptr<carryType>(new carryType());
	xGrad = shared_ptr<carryType>(new carryType());
	yGrad = shared_ptr<carryType>(new carryType());
}
void bias_bpg::fwd() {
	bias_forward(x, y, *param);
}
void bias_bpg::bwd() {
	bias_backward(yGrad, xGrad, *param);
}
model* bias_bpg::clone() {
	bias_bpg* result = new bias_bpg(*this);
	return result;
}
activate::~activate() {
	activate_deconstruct(act);
}
activate::activate(activation* act) {
	x = shared_ptr<carryType>(new carryType());
	y = shared_ptr<carryType>(new carryType());
	this->act = act;
}
void activate::fwd() {
	activate_forward(x, y, act);
}
model* activate::clone() {
	return new activate(*this);
}
activate_bpg::~activate_bpg() {
	activate_deconstruct(act);
}
activate_bpg::activate_bpg(activation* act) {
	x = shared_ptr<carryType>(new carryType());
	y = shared_ptr<carryType>(new carryType());
	xGrad = shared_ptr<carryType>(new carryType());
	yGrad = shared_ptr<carryType>(new carryType());
	this->act = act;
}
void activate_bpg::fwd() {
	activate_forward(x, y, act);
}
void activate_bpg::bwd() {
	activate_backward(yGrad, y, xGrad, act);
}
model* activate_bpg::clone() {
	return new activate_bpg(*this);
}
weight::~weight() {
	weight_deconstruct(param);
}
weight::weight() {
	x = shared_ptr<carryType>(new carryType());
	y = shared_ptr<carryType>(new carryType());
}
void weight::fwd() {
	weight_forward(x, y, *param);
}
model* weight::clone() {
	weight* result = new weight(*this);
	return result;
}
weight_bpg::~weight_bpg() {
	weight_deconstruct(param);
}
weight_bpg::weight_bpg() {
	x = shared_ptr<carryType>(new carryType());
	y = shared_ptr<carryType>(new carryType());
	xGrad = shared_ptr<carryType>(new carryType());
	yGrad = shared_ptr<carryType>(new carryType());
}
void weight_bpg::fwd() {
	weight_forward(x, y, *param);
}
void weight_bpg::bwd() {
	weight_backward(yGrad, xGrad, x, *param);
}
model* weight_bpg::clone() {
	weight_bpg* result = new weight_bpg(*this);
	return result;
}
weightSet::~weightSet() {
	weightSet_deconstruct(this);
}
weightSet::weightSet(int a) {
	x = shared_ptr<carryType>(new carryType());
	y = shared_ptr<carryType>(new carryType({}));
	for (int i = 0; i < a; i++) {
		y->value_vector.push_back(shared_ptr<carryType>(new carryType()));
		push_back(new weight());
	}
}
void weightSet::fwd() {
	weightSet_forward(x, y, this);
}
void weightSet::foreach(function<void(model*)> func) {
	weightSet_foreach(func, this, this);
}
model* weightSet::clone() {
	weightSet* result = new weightSet(size());
	for (int i = 0; i < size(); i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
weightSet_bpg::~weightSet_bpg() {
	weightSet_deconstruct(this);
}
weightSet_bpg::weightSet_bpg(int a) {
	x = shared_ptr<carryType>(new carryType());
	y = shared_ptr<carryType>(new carryType({}));
	xGrad = shared_ptr<carryType>(new carryType());
	yGrad = shared_ptr<carryType>(new carryType({}));
	for (int i = 0; i < a; i++) {
		y->value_vector.push_back(shared_ptr<carryType>(new carryType()));
		yGrad->value_vector.push_back(shared_ptr<carryType>(new carryType()));
		push_back(new weight_bpg());
	}
}
void weightSet_bpg::fwd() {
	weightSet_forward(x, y, this);
}
void weightSet_bpg::bwd() {
	weightSet_backward(yGrad, xGrad, this);
}
void weightSet_bpg::foreach(function<void(model*)> func) {
	weightSet_foreach(func, this, this);
}
model* weightSet_bpg::clone() {
	weightSet_bpg* result = new weightSet_bpg(size());
	for (int i = 0; i < size(); i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
weightJunction::~weightJunction() {
	weightJunction_deconstruct(this);
}
weightJunction::weightJunction(int a, int b) {
	this->a = a;
	this->b = b;
	x = shared_ptr<carryType>(new carryType({}));
	y = shared_ptr<carryType>(new carryType({}));
	for (int i = 0; i < a; i++) {
		x->value_vector.push_back(shared_ptr<carryType>(new carryType()));
		push_back(new weightSet(b));
	}
	for (int i = 0; i < b; i++) {
		y->value_vector.push_back(shared_ptr<carryType>(new carryType()));
	}
}
void weightJunction::fwd() {
	weightJunction_forward(x, y, &b, this);
}
void weightJunction::foreach(function<void(model*)> func) {
	weightJunction_foreach(func, this, this);
}
model* weightJunction::clone() {
	weightJunction* result = new weightJunction(size(), ((vector<model*>*)at(0))->size());
	for (int i = 0; i < size(); i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
weightJunction_bpg::~weightJunction_bpg() {
	weightJunction_deconstruct(this);
}
weightJunction_bpg::weightJunction_bpg(int a, int b) {
	this->a = a;
	this->b = b;
	x = shared_ptr<carryType>(new carryType({}));
	xGrad = shared_ptr<carryType>(new carryType({}));
	y = shared_ptr<carryType>(new carryType({}));
	yGrad = shared_ptr<carryType>(new carryType({}));
	for (int i = 0; i < a; i++) {
		x->value_vector.push_back(shared_ptr<carryType>(new carryType()));
		xGrad->value_vector.push_back(shared_ptr<carryType>(new carryType()));
		push_back(new weightSet_bpg(b));
	}
	for (int i = 0; i < b; i++) {
		y->value_vector.push_back(shared_ptr<carryType>(new carryType()));
		yGrad->value_vector.push_back(shared_ptr<carryType>(new carryType()));
	}
}
void weightJunction_bpg::fwd() {
	weightJunction_forward(x, y, &b, this);
}
void weightJunction_bpg::bwd() {
	weightJunction_backward(yGrad, xGrad, this);
}
void weightJunction_bpg::foreach(function<void(model*)> func) {
	weightJunction_foreach(func, this, this);
}
model* weightJunction_bpg::clone() {
	weightJunction_bpg* result = new weightJunction_bpg(size(), ((vector<model*>*)at(0))->size());
	for (int i = 0; i < size(); i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
sequential::~sequential() {
	sequential_deconstruct(this);
}
sequential::sequential() {
	x = shared_ptr<carryType>(new carryType());
	y = shared_ptr<carryType>(new carryType());
}
void sequential::fwd() {
	sequential_forward(x, y, this);
}
void sequential::foreach(function<void(model*)> func) {
	sequential_foreach(func, this, this);
}
model* sequential::clone() {
	sequential* result = new sequential();
	for (int i = 0; i < size(); i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
sequential_bpg::~sequential_bpg() {
	sequential_deconstruct(this);
}
sequential_bpg::sequential_bpg() {
	x = shared_ptr<carryType>(new carryType());
	y = shared_ptr<carryType>(new carryType());
	xGrad = shared_ptr<carryType>(new carryType());
	yGrad = shared_ptr<carryType>(new carryType());
}
void sequential_bpg::fwd() {
	sequential_forward(x, y, this);
}
void sequential_bpg::bwd() {
	sequential_backward(yGrad, xGrad, this);
}
void sequential_bpg::foreach(function<void(model*)> func) {
	sequential_foreach(func, this, this);
}
model* sequential_bpg::clone() {
	sequential_bpg* result = new sequential_bpg();
	for (int i = 0; i < size(); i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
layer::~layer() {
	layer_deconstruct(this);
}
layer::layer() {
	x = shared_ptr<carryType>(new carryType({}));
	y = shared_ptr<carryType>(new carryType({}));
}
layer::layer(int a, model* model_default) {
	x = shared_ptr<carryType>(new carryType({}));
	y = shared_ptr<carryType>(new carryType({}));
	for (int i = 0; i < a; i++) {
		x->value_vector.push_back(shared_ptr<carryType>(new carryType()));
		y->value_vector.push_back(shared_ptr<carryType>(new carryType()));
		push_back(model_default->clone());
	}
}
void layer::fwd() {
	layer_forward(x, y, this);
}
void layer::foreach(function<void(model*)> func) {
	layer_foreach(func, this, this);
}
model* layer::clone() {
	layer* result = new layer();
	for (int i = 0; i < size(); i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}
layer_bpg::~layer_bpg() {
	layer_deconstruct(this);
}
layer_bpg::layer_bpg() {
	x = shared_ptr<carryType>(new carryType({}));
	y = shared_ptr<carryType>(new carryType({}));
	xGrad = shared_ptr<carryType>(new carryType({}));
	yGrad = shared_ptr<carryType>(new carryType({}));
}
layer_bpg::layer_bpg(int a, model* model_default) {
	x = shared_ptr<carryType>(new carryType({}));
	y = shared_ptr<carryType>(new carryType({}));
	xGrad = shared_ptr<carryType>(new carryType({}));
	yGrad = shared_ptr<carryType>(new carryType({}));
	for (int i = 0; i < a; i++) {
		x->value_vector.push_back(shared_ptr<carryType>(new carryType()));
		y->value_vector.push_back(shared_ptr<carryType>(new carryType()));
		xGrad->value_vector.push_back(shared_ptr<carryType>(new carryType()));
		yGrad->value_vector.push_back(shared_ptr<carryType>(new carryType()));
		push_back(model_default->clone());
	}
}
void layer_bpg::fwd() {
	layer_forward(x, y, this);
}
void layer_bpg::bwd() {
	layer_backward(yGrad, xGrad, this);
}
void layer_bpg::foreach(function<void(model*)> func) {
	layer_foreach(func, this, this);
}
model* layer_bpg::clone() {
	layer_bpg* result = new layer_bpg();
	for (int i = 0; i < size(); i++) {
		result->push_back(at(i)->clone());
	}
	return result;
}