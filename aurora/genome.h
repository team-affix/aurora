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
			genome(
				const aurora::maths::tensor& a_alleles,
				const std::function<double(double)>& a_random_change
			);
			operator aurora::maths::tensor& ();

		public:
			genome mutate() const;
			std::vector<genome> mutate(
				const size_t& a_children
			);
			static std::vector<genome> mutate(
				const std::vector<genome>& a_genomes
			);
			
		public:
			genome merge(
				genome& a_spouse
			);
			std::vector<genome> merge(
				genome& a_spouse,
				const size_t& a_children
			);
			static genome merge(
				std::vector<genome> a_parents
			);
			static std::vector<genome> merge(
				std::vector<genome> a_parents,
				const size_t& a_children
			);

		public:
			genome clone() const;

		};

	}
}
