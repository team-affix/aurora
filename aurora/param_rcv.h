#pragma once
#include "pch.h"
#include "param.h"

using aurora::params::param;

namespace aurora {
	namespace params {
		class param_rcv : public param {
		public:
			ptr<double> m_dstate = new double(0);
			ptr<double> m_learn_rate = new double(0);
			ptr<double> m_reward = new double(0);
			ptr<double> m_dreward = new double(0);
			ptr<double> m_beta = new double(0);
			ptr<double> m_alpha = new double(0);
			ptr<double> m_momentum = new double(0);

		public:
			virtual ~param_rcv();
			param_rcv();
			param_rcv(const double& a_state, const double& a_learn_rate, const double& a_beta);

		public:
			double sign(const double& a_x);
			double& dstate();
			double& reward();
			double& dreward();
			double& learn_rate();
			const double& beta();
			double& momentum();
			const double& alpha();

		public:
			void beta(const double& a_val);

		public:
			void update(const double& a_c);
			void reward(const double& a_reward);
			void dreward(const double& a_dreward);

		};
		typedef ptr<param_rcv> Param_rcv;
	}
}
