#pragma once
#include "superHeader.h"
#include "maths.h"

namespace optimization {
	class parameter;
	class parameter_mutation;
	class parameter_sgd;
	class parameter_momentum;
	class optimizer;
	class optimizer_mutation;
	class optimizer_sgd;
	class optimizer_momentum;

	class parameter {
	public:
		double state;
		double learnRate;
	};
	class parameter_mutation : public parameter {
	public:
		double rcv;
		double state_previous;
		double momentum;
		double beta;
	};
	class parameter_sgd : public parameter {
	public:
		double gradient;
	};
	class parameter_momentum : public parameter_sgd {
	public:
		double momentum;
		double beta;
	};
	class optimizer {
	};
	class optimizer_mutation {
	public:
		virtual void update_state(parameter_mutation* p, int domainSize);
		virtual void roll_back_state(parameter_mutation* p);
		virtual void keep_state(parameter_mutation* p);
	};
	class optimizer_sgd {
	public:
		virtual void update_state(parameter_sgd* p);
		virtual void clear_gradient(parameter_sgd* p);
	};
	class optimizer_momentum : public optimizer_sgd {
	public:
		virtual void update_state(parameter_momentum* p);
		virtual void update_momentum(parameter_momentum* p);
	};
}