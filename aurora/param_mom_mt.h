#pragma once
#include "affix-base/pch.h"
#include "param_mom.h"

namespace aurora {
	namespace params {
		class param_mom_mt : public param_mom {
		public:
			affix_base::data::ptr<double> m_momentum_ptr = new double(0);
			affix_base::data::ptr<double> m_beta_ptr = new double(0);

		public:
			std::mutex m_accum_grad_mtx;

		public:
			virtual ~param_mom_mt();
			param_mom_mt();
			param_mom_mt(
				double a_state,
				double a_learn_rate,
				double a_gradient,
				double a_momentum,
				double a_beta
			);

		public:
			virtual void accum_grad(
				double a_grad
			);

		public:
			virtual param* clone();

		};
		typedef affix_base::data::ptr<param_mom_mt> Param_mom_mt;
	}
}
