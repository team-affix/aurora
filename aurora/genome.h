#pragma once
#include "pch.h"
#include "tensor.h"

using aurora::maths::tensor;
using std::function;

namespace aurora {
	namespace evolution {
		class genome : public tensor {
		public:
			function<double(double)> random_change;

		public:
			genome();
			genome(tensor a_alleles, function<double(double)> a_random_change);
			operator tensor& ();

		public:
			genome mutate();
			vector<genome> mutate(size_t a_children);
			static vector<genome> mutate(vector<genome> a_genomes);
			
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