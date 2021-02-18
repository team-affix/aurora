#include "generation.h"
#include "sort.h"

using namespace aurora::pseudo;
using aurora::evolution::generation;

generation::generation() {

}

generation::generation(vector<genome> a_genomes, function<double(genome&)> a_get_reward) {
	this->genomes = a_genomes;
	this->get_reward = a_get_reward;
}

genome& generation::best() {
	return genomes[sort().front()];
}

vector<genome> generation::best(size_t a_genomes) {
	vector<genome> result = vector<genome>(a_genomes);
	vector<size_t> sorted = sort();
	for (int i = 0; i < a_genomes; i++)
		result[i] = genomes[sorted[i]];
	return result;
}

genome& generation::worst() {
	return genomes[sort().back()];
}

vector<genome> generation::worst(size_t a_genomes) {
	vector<genome> result = vector<genome>(a_genomes);
	vector<size_t> sorted = sort();
	for (int i = 0; i < a_genomes; i++)
		result[i] = genomes[sorted[sorted.size() - 1 - i]];
	return result;
}

vector<size_t> generation::sort() {
	vector<size_t> result = vector<size_t>(genomes.size());
	vector<double> rewards = vector<double>(genomes.size());
	for (int i = 0; i < genomes.size(); i++) {
		double reward = get_reward(genomes[i]);
		size_t insert_index = gtl_insertion_index(rewards, reward);
		result.insert(result.begin() + insert_index, i);
		rewards.insert(rewards.begin() + insert_index, reward);
	}
	return result;
}