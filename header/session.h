#pragma once
#include "pch.h"

using std::function;

namespace aurora {
	namespace training {
		class session {
		public:
			~session();
			session();

		private:
			void init();

		public:
			function<void(size_t)> fit_once;


		};
	}
}