#include "genome.h"

using aurora::evolution::genome;

genome::genome() {

}

genome::genome(tensor a_alleles, double a_mut_prob, double a_learn_rate) : tensor(a_alleles) {
	this->mut_prob = a_mut_prob;
	this->learn_rate = a_learn_rate;
}

genome::operator tensor& () {
	return *this;
}

genome genome::mutate(default_random_engine& a_re) {
	genome result = clone();
	int divisor = 1 / mut_prob;
	for (int allele = 0; allele < result.size(); allele++)
		if (rand() % divisor == 0)
			result[allele].val() += learn_rate * s_urd(a_re);
	return result;
}

vector<genome> genome::mutate(default_random_engine& a_re, size_t a_children) {
	vector<genome> result = vector<genome>(a_children);
	for (int child = 0; child < a_children; child++)
		result[child] = mutate(a_re);
	return result;
}

vector<genome> genome::mutate(vector<genome> a_genomes, default_random_engine& a_re) {
	vector<genome> result = vector<genome>(a_genomes);
	for (int i = 0; i < result.size(); i++)
		result[i] = a_genomes[i].mutate(a_re);
	return result;
}

genome genome::merge(genome& a_spouse) {
	genome result = clone();
	for (int allele = 0; allele < result.size(); allele++)
		if (rand() % 2 == 0)
			result[allele].val() = a_spouse[allele].val();
	return result;
}

vector<genome> genome::merge(genome& a_spouse, size_t a_children) {
	vector<genome> result = vector<genome>(a_children);
	for (int child = 0; child < a_children; child++)
		result[child] = merge(a_spouse);
	return result;
}

genome genome::merge(vector<genome> a_parents) {
	assert(a_parents.size() > 0);
	genome result = a_parents.front().clone();
	for (int allele = 0; allele < result.size(); allele++) {
		size_t parent_index = rand() % a_parents.size();
		result[allele].val() = a_parents[parent_index][allele].val();
	}
	return result;
}

vector<genome> genome::merge(vector<genome> a_parents, size_t a_children) {
	vector<genome> result = vector<genome>(a_children);
	for (int child = 0; child < a_children; child++)
		result[child] = merge(a_parents);
	return result;
}

genome genome::clone() {
	return genome(tensor::clone(), mut_prob, learn_rate);
}