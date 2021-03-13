#include "genome.h"

using aurora::evolution::genome;

genome::genome() {

}

genome::genome(tensor a_alleles, function<double(double)> a_random_change) : tensor(a_alleles) {
	this->random_change = a_random_change;
}

genome::operator tensor& () {
	return *this;
}

genome genome::mutate() {
	genome result = clone();
	for (int allele = 0; allele < result.size(); allele++)
		result[allele].val() = random_change(at(allele));
	return result;
}

vector<genome> genome::mutate(size_t a_children) {
	vector<genome> result = vector<genome>(a_children);
	for (int child = 0; child < a_children; child++)
		result[child] = mutate();
	return result;
}

vector<genome> genome::mutate(vector<genome> a_genomes) {
	vector<genome> result = vector<genome>(a_genomes);
	for (int i = 0; i < result.size(); i++)
		result[i] = a_genomes[i].mutate();
	return result;
}

genome genome::merge(genome& a_spouse) {
	genome result = clone();
	size_t slice_index = rand() % size();
	result.range(0, slice_index).pop(a_spouse.range(0, slice_index));
	return result;
}

vector<genome> genome::merge(genome& a_spouse, size_t a_children) {
	vector<genome> result = vector<genome>(a_children);
	for (int child = 0; child < a_children; child++)
		result[child] = merge(a_spouse);
	return result;
}

genome genome::merge(vector<genome> a_parents) {
	assert(a_parents.size() >= 2);
	genome result = a_parents.front().clone();
	for (int parent = 1; parent < a_parents.size(); parent++)
		result = result.merge(a_parents[parent]);
	return result;
}

vector<genome> genome::merge(vector<genome> a_parents, size_t a_children) {
	vector<genome> result = vector<genome>(a_children);
	for (int child = 0; child < a_children; child++)
		result[child] = merge(a_parents);
	return result;
}

genome genome::clone() {
	return genome(tensor::clone(), random_change);
}