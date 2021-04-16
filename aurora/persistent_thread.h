#pragma once
#include "pch.h"
#include "ptr.h"

using aurora::data::ptr;
using std::function;
using std::vector;

namespace aurora {
	namespace pseudo {
		class persistent_thread {
		private:
			volatile bool active = true;
			std::thread thd;
			
		private:
			volatile bool executing = false;
			function<void()> func;

		private:
			virtual void loop();

		public:
			virtual ~persistent_thread();
			persistent_thread();

		public:
			virtual void execute(function<void()> a_func);
			virtual void join();

		};
	}
}