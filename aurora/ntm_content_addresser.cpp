#include "pch.h"
#include "ntm_content_addresser.h"

using aurora::models::ntm_content_addresser;

ntm_content_addresser::~ntm_content_addresser() {

}

ntm_content_addresser::ntm_content_addresser() {

}

ntm_content_addresser::ntm_content_addresser(size_t a_memory_height, size_t a_memory_width) {
	memory_height = a_memory_height;
	memory_width = a_memory_width;
	internal_similarity = new sync(new cos_sim(a_memory_width));
	internal_sparsify = new ntm_sparsify(memory_height);
	internal_normalize = new normalize(memory_height);
}

void ntm_content_addresser::param_recur(function<void(Param&)> a_func) {

}

model* ntm_content_addresser::clone(function<Param(Param&)> a_func) {
	ntm_content_addresser* result = new ntm_content_addresser();
	result->memory_height = memory_height;
	result->memory_width = memory_width;
	result->internal_similarity = (sync*)internal_similarity->clone(a_func);
	result->internal_sparsify = (ntm_sparsify*)internal_sparsify->clone(a_func);
	result->internal_normalize = (normalize*)internal_normalize->clone(a_func);
	return result;
}

void ntm_content_addresser::fwd() {
	internal_similarity->fwd();
	internal_sparsify->fwd();
	internal_normalize->fwd();
}

void ntm_content_addresser::bwd() {
	internal_normalize->bwd();
	internal_sparsify->bwd();
	internal_similarity->bwd();

	key_grad.clear();
	for (int i = 0; i < memory_height; i++)
		key_grad.add_1d(internal_similarity->x_grad[i][0], key_grad);
}

void ntm_content_addresser::signal(tensor& a_y_des) {
	y.sub_1d(a_y_des, y_grad);
}

void ntm_content_addresser::model_recur(function<void(model*)> a_func) {
	a_func(this);
	internal_similarity->model_recur(a_func);
	internal_sparsify->model_recur(a_func);
	internal_normalize->model_recur(a_func);
}

void ntm_content_addresser::compile() {
	x = tensor::new_2d(memory_height, memory_width);
	x_grad = tensor::new_2d(memory_height, memory_width);
	y = tensor::new_1d(memory_height);
	y_grad = tensor::new_1d(memory_height);
	key = tensor::new_1d(memory_width);
	key_grad = tensor::new_1d(memory_width);
	beta = tensor::new_1d(1);
	beta_grad = tensor::new_1d(1);

	internal_similarity->prep(memory_height);
	internal_similarity->compile();
	internal_similarity->unroll(memory_height);

	internal_sparsify->compile();
	internal_normalize->compile();

	for (int i = 0; i < memory_height; i++) {
		key.group_join(internal_similarity->x[i][0]);
		x[i].group_join(internal_similarity->x[i][1]);
		x_grad[i].group_join(internal_similarity->x_grad[i][1]);
	}

	internal_similarity->y.group_join(internal_sparsify->x);
	internal_similarity->y_grad.group_join(internal_sparsify->x_grad);

	internal_sparsify->y.group_join(internal_normalize->x);
	internal_sparsify->y_grad.group_join(internal_normalize->x_grad);

	internal_normalize->y.group_join(y);
	internal_normalize->y_grad.group_join(y_grad);

	beta.group_join(internal_sparsify->beta);
	beta_grad.group_join(internal_sparsify->beta_grad);

}