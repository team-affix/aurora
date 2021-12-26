#pragma once
#include "affix-base/pch.h"
#include "tensor.h"


namespace aurora {
	namespace evolution {
		class genome : public aurora::maths::tensor {
		public:
			std::function<double(double)> m_random_change;

		public:
			genome();
			genome(aurora::maths::tensor a_alleles, std::function<double(double)> a_random_change);
			operator aurora::maths::tensor& ();

		public:
			genome mutate();
			std::vector<genome> mutate(size_t a_children);
			static std::vector<genome> mutate(std::vector<genome> a_genomes);
			
		public:
			genome merge(genome& a_spouse);
			std::vector<genome> merge(genome& a_spouse, size_t a_children);
			static genome merge(std::vector<genome> a_parents);
			static std::vector<genome> merge(std::vector<genome> a_parents, size_t a_children);

		public:
			genome clone();

		};

	}
}
