#pragma once
#include "model.h"
#include "loss.h"
#include "tensor.h"

namespace aurora
{
	namespace models
	{
		class mse_loss : public loss
		{
		public:
			double m_y_lowest_rank_count = 0;
			double m_2_over_y_lowest_rank_count = 0;

		public:
			Model m_model;

		public:
			MODEL_FIELDS
			virtual ~mse_loss();
			mse_loss();
			mse_loss(
				const Model& a_model
			);

		public:
			virtual double signal(
				const aurora::maths::tensor& a_y_des,
				const aurora::maths::tensor* a_y = nullptr,
				aurora::maths::tensor* a_y_grad = nullptr
			);
			virtual double cycle(
				const aurora::maths::tensor& a_x,
				const aurora::maths::tensor& a_y_des
			);

		};
		typedef affix_base::data::ptr<mse_loss> Mse_loss;
	}
}
