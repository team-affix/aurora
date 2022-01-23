#include "affix-base/pch.h"
#include "generation.h"
#include "affix-base/insertion_index.h"

using namespace affix_base::sorting;
using aurora::evolution::generation;

generation::generation() {

}

generation::generation(
	const vector<genome>& a_genomes,
	const std::function<double(genome&)>& a_get_reward
) {
	this->m_genomes = a_genomes;
	this->m_get_reward = a_get_reward;
}

aurora::evolution::genome& generation::best() {
	return m_genomes[sort().front()];
}

vector<aurora::evolution::genome> generation::best(
	const size_t& a_genomes
)
{
	vector<genome> result = vector<genome>(a_genomes);
	vector<size_t> sorted = sort();
	for (int i = 0; i < a_genomes; i++)
		result[i] = m_genomes[sorted[i]];
	return result;
}

aurora::evolution::genome& generation::worst() {
	return m_genomes[sort().back()];
}

vector<aurora::evolution::genome> generation::worst(
	const size_t& a_genomes
)
{
	vector<genome> result = vector<genome>(a_genomes);
	vector<size_t> sorted = sort();
	for (int i = 0; i < a_genomes; i++)
		result[i] = m_genomes[sorted[sorted.size() - 1 - i]];
	return result;
}

vector<size_t> generation::sort() {
	vector<size_t> result = vector<size_t>(m_genomes.size());
	vector<double> rewards = vector<double>(m_genomes.size());
	for (int i = 0; i < m_genomes.size(); i++) {
		double reward = m_get_reward(m_genomes[i]);
		size_t insert_index = gtl_insertion_index(rewards, reward);
		result.insert(result.begin() + insert_index, i);
		rewards.insert(rewards.begin() + insert_index, reward);
	}
	return result;
}