#pragma once
#include "affix-base/pch.h"
#include "param.h"
#include "param_sgd.h"
#include "param_mom.h"
#include "param_sgd_mt.h"
#include "param_mom_mt.h"

namespace aurora {
	namespace pseudo {

		#define PARAM_INIT(right_side, param_vec) \
		[&](aurora::params::Param& pmt) { \
		pmt = new right_side; \
		param_vec.push_back(pmt); \
		}

		#define PARAM_DUMP(param_vec) \
		[&](aurora::params::Param& pmt) { \
		param_vec.push_back(pmt.get()); \
		}

		#define PARAM_COUNT(incrementer) \
		[&](aurora::params::Param& pmt) { \
			incrementer++; \
		} \


	}
}
