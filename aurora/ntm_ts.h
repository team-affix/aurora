#pragma once
#include "pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class ntm_ts : public model {
		public:
			size_t units;
			size_t reads;
			size_t writes;

		public:
			tensor 
			
		public:
			MODEL_FIELDS
			virtual ~ntm_ts();
			ntm_ts();

		};
	}
}