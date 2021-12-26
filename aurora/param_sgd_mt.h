#pragma once
#include "affix-base/pch.h"
#include "param_sgd.h"

namespace aurora {
	namespace params {
		class param_sgd_mt : public param_sgd {
		public:
			affix_base::data::ptr<double> m_gradient_ptr = new double(0);

		public:
			std::mutex m_accum_grad_mutex;

		public:
			virtual ~param_sgd_mt();
			param_sgd_mt();
			param_sgd_mt(
				double a_state,
				double a_learn_rate,
				double a_gradient
			);

		public:
			virtual void accum_grad(
				double a_grad
			);

		public:
			virtual param* clone();

		};
		typedef affix_base::data::ptr<param_sgd_mt> Param_sgd_mt;
	}
}
