#include "affix-base/pch.h"
#include "genome.h"

using aurora::evolution::genome;

genome::genome() {

}

genome::genome(aurora::maths::tensor a_alleles, std::function<double(double)> a_random_change) : aurora::maths::tensor(a_alleles) {
	this->m_random_change = a_random_change;
}

genome::operator aurora::maths::tensor& () {
	return *this;
}

genome genome::mutate() {
	genome result = clone();
	for (int allele = 0; allele < result.size(); allele++)
		result[allele].val() = m_random_change(at(allele));
	return result;
}

std::vector<genome> genome::mutate(size_t a_children) {
	std::vector<genome> result = std::vector<genome>(a_children);
	for (int child = 0; child < a_children; child++)
		result[child] = mutate();
	return result;
}

std::vector<genome> genome::mutate(std::vector<genome> a_genomes) {
	std::vector<genome> result = std::vector<genome>(a_genomes);
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

std::vector<genome> genome::merge(genome& a_spouse, size_t a_children) {
	std::vector<genome> result = std::vector<genome>(a_children);
	for (int child = 0; child < a_children; child++)
		result[child] = merge(a_spouse);
	return result;
}

genome genome::merge(std::vector<genome> a_parents) {
	assert(a_parents.size() >= 2);
	genome result = a_parents.front().clone();
	for (int parent = 1; parent < a_parents.size(); parent++)
		result = result.merge(a_parents[parent]);
	return result;
}

std::vector<genome> genome::merge(std::vector<genome> a_parents, size_t a_children) {
	std::vector<genome> result = std::vector<genome>(a_children);
	for (int child = 0; child < a_children; child++)
		result[child] = merge(a_parents);
	return result;
}

genome genome::clone() {
	return genome(tensor::clone(), m_random_change);
}