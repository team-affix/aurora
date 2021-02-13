#pragma once
#include "genome.h"

using aurora::optimization::genome;

namespace aurora {
	namespace optimization {
		class generation {
		public:
			genome parent;
			vector<genome> genomes;

		public:
			size_t best_child_index();
			genome& best_child();
			double cycle_generation();

		};
	}
}