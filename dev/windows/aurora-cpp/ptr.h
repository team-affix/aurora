#pragma once
#include <memory>

using std::shared_ptr;

namespace aurora {
	namespace data {
		template<class T>
		class ptr : public shared_ptr<T> {
		public:
			ptr() {
			
			}
			ptr(T* a) {
				shared_ptr<T>::reset(a);
			}

		public:
			T& val() {
				return *shared_ptr<T>::get();
			}

		public:
			void operator=(T a) {
				*shared_ptr<T>::get() = a;
			}

		public:
			void link(ptr<T>& other) {
				shared_ptr<T>::reset(other.get());
			}

		};
	}
}