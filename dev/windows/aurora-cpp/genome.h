#pragma once
#include "tensor.h"
#include <assert.h>

using aurora::math::tensor;

namespace aurora {
	namespace evolution {
		static uniform_real_distribution<double> s_urd(-1, 1);
		class genome : public tensor {
		public:
			double mut_prob;
			double learn_rate;

		public:
			genome();
			genome(tensor a_alleles, double a_mut_prob, double a_learn_rate);
			operator tensor& ();

		public:
			genome mutate(default_random_engine& a_re);
			vector<genome> mutate(default_random_engine& a_re, size_t a_children);
			static vector<genome> mutate(vector<genome> a_genomes, default_random_engine& a_re);
			
		public:
			genome merge(genome& a_spouse);
			vector<genome> merge(genome& a_spouse, size_t a_children);
			static genome merge(vector<genome> a_parents);
			static vector<genome> merge(vector<genome> a_parents, size_t a_children);

		public:
			genome clone();

		};

	}
}