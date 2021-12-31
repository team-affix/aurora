#pragma once
#include "loss.h"

namespace aurora
{
	namespace models
	{
		class ce_loss : public loss
		{
		public:
			Model m_model;

		public:
			MODEL_FIELDS
			virtual ~ce_loss();
			ce_loss();
			ce_loss(
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
		typedef affix_base::data::ptr<ce_loss> Ce_loss;
	}
}
