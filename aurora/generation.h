#pragma once
#include "affix-base/pch.h"
#include "genome.h"

namespace aurora {
	namespace evolution {
		class generation {
		public:
			std::vector<genome> genomes;
			std::function<double(genome&)> get_reward;

		public:
			generation();
			generation(std::vector<genome> a_genomes, std::function<double(genome&)> a_get_reward);

		public:
			genome& best();
			std::vector<genome> best(size_t a_genomes);
			genome& worst();
			std::vector<genome> worst(size_t a_genomes);
			std::vector<size_t> sort();

		};
	}
}
