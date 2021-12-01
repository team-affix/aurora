#pragma once
#include "pch.h"
#include "param.h"

using aurora::params::param;
using std::uniform_real_distribution;
using std::normal_distribution;

namespace aurora {
	namespace params {
		class param_rcv : public param {
		protected:
			static uniform_real_distribution<double> m_percentage_urd;
			static normal_distribution<double> m_rcv_nd;
			static uniform_real_distribution<double> m_rcv_urd;

		public:
			ptr<double> m_s_prev = new double(0);			// STATE AT PREVIOUS EPOCH
			ptr<double> m_learn_rate = new double(0);		// LEARN RATE OF PARAMETER
			ptr<double> m_l_prev = new double(0);			// LOSS AT PREVIOUS EPOCH
			ptr<double> m_beta = new double(0);				// BETA
			ptr<double> m_alpha = new double(0);			// 1 - BETA
			ptr<double> m_running_average = new double(0);	// MOMENTUM
			ptr<double> m_slope = new double(0);			// IRC
			ptr<double> m_slope_rs = new double(0);			// ROOT-SQUARE OF ALL PARAMETERS' SLOPES

		public:
			virtual ~param_rcv();
			param_rcv();
			param_rcv(const double& a_state, const double& a_learn_rate, const double& a_beta);

		public:
			double& learn_rate();
			const double& beta();
			const double& alpha();
			double& running_average();
			double& slope();
			double& slope_rs();

		public:
			double sign(const double& a_x);

		public:
			void beta(const double& a_val);

		public:
			void signal(const double& a_loss);
			void update();

		};
		typedef ptr<param_rcv> Param_rcv;
	}
}
