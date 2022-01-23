#pragma once
#include "affix-base/pch.h"
#include "genome.h"

namespace aurora {
	namespace evolution {
		class generation {
		public:
			std::vector<genome> m_genomes;
			std::function<double(genome&)> m_get_reward;

		public:
			generation();
			generation(
				const std::vector<genome>& a_genomes,
				const std::function<double(genome&)>& a_get_reward
			);

		public:
			genome& best();
			std::vector<genome> best(
				const size_t& a_genomes
			);
			genome& worst();
			std::vector<genome> worst(
				const size_t& a_genomes
			);
			std::vector<size_t> sort();

		};
	}
}
