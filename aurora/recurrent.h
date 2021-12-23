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
		typedef ptr<recurrent> Recurrent;
	}
}
