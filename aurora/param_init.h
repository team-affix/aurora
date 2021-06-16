#pragma once
#include "pch.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"
#include "param_sgd_mt.h"
#include "param_mom_mt.h"

using namespace aurora::params;
using std::uniform_real_distribution;

namespace aurora {
	namespace pseudo {


		#define PARAM_INIT(right_side, param_vector) \
		[&](Param& pmt) { \
		auto l_pmt = new right_side; \
		pmt = l_pmt; \
		param_vector.push_back(l_pmt); \
		}

		#define PARAM_DUMP(param_vector) \
		[&](Param& pmt) { \
		param_vector.push_back(pmt.get()); \
		}

		#define PARAM_COUNT(incrementer) \
		[&](Param& pmt) { \
			incrementer++; \
		} \


	}
}