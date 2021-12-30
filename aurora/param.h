#pragma once
#include "affix-base/pch.h"
#include "affix-base/ptr.h"

namespace aurora {
	namespace params {
		class param_sgd;
		class param_mom;
		class param {
		public:
			affix_base::data::ptr<double> m_state_ptr = new double(0);

		public:
			virtual ~param();
			param();
			param(
				double a_state
			);

		public:
			virtual double& state();
			virtual const double& state() const;

		public:
			virtual void update();
			virtual param* clone() const;

		};
		typedef affix_base::data::ptr<param> Param;
	}
}
