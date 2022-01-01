#include "affix-base/pch.h"
#include "model.h"

using namespace aurora;
using namespace models;
using std::function;
using aurora::params::Param;
using aurora::models::model;
using aurora::params::param_sgd;
using aurora::maths::tensor;
using std::vector;

tensor& model::x()
{
	return m_x;
}

tensor& model::y()
{
	return m_y;
}

tensor& model::x_grad()
{
	return m_x_grad;
}

tensor& model::y_grad()
{
	return m_y_grad;
}

model::~model() {

}

model::model() {

}

void model::fwd() {

}

void model::bwd() {

}

void model::model_recur(const function<void(model*)>& a_func) {
	a_func(this);
}

void model::param_recur(const function<void(Param&)>& a_func) {

}

model* model::clone(const function<Param(Param&)>& a_func) {
	return new model();
}

void model::compile() {
	m_x.link(m_y);
	m_x_grad.link(m_y_grad);
}
