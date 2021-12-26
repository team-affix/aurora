#pragma once
#include "affix-base/pch.h"
#include "recurrent.h"

namespace aurora {
	namespace models {
		class stacked_recurrent : public recurrent {
		public:
			size_t m_prepared_size = 0;
			size_t m_unrolled_size = 0;

		public:
			std::vector<Recurrent> m_models;

		public:
			RECURRENT_FIELDS
			virtual ~stacked_recurrent();
			stacked_recurrent();
			stacked_recurrent(
				std::vector<Recurrent> a_models
			);
			stacked_recurrent(
				size_t a_height,
				Recurrent a_model_template
			);

		};
		typedef affix_base::data::ptr<stacked_recurrent> Stacked_recurrent;
	}
}
