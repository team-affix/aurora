#pragma once
#include "genome.h"

using aurora::evolution::genome;

namespace aurora {
	namespace evolution {
		class generation {
		public:
			vector<genome> genomes;
			function<double(genome&)> get_reward;

		public:
			generation();
			generation(vector<genome> a_genomes, function<double(genome&)> a_get_reward);

		public:
			genome& best();
			vector<genome> best(size_t a_genomes);
			genome& worst();
			vector<genome> worst(size_t a_genomes);
			vector<size_t> sort();

		};
	}
}