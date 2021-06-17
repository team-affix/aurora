#pragma once
#include "pch.h"
#include "model.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class recurrent : public model {
		public:
			RECURRENT_FIELDS
			virtual ~recurrent();
			recurrent();
			
		};
	}
}
