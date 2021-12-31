#pragma once
#include "model.h"

namespace aurora
{
	namespace models
	{
		class loss : public model
		{
		public:
			Model m_model;

		public:
			MODEL_FIELDS
			virtual ~loss();
			loss();
			loss(
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
	}
}
