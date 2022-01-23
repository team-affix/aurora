#pragma once
#include "affix-base/pch.h"
#include "model.h"

namespace aurora {
	namespace models {
		class recurrent : public model {
		public:
			RECURRENT_FIELDS
			virtual ~recurrent();
			recurrent();
			
		};
		typedef affix_base::data::ptr<recurrent> Recurrent;
	}
}
