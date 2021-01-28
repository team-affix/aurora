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
			ptr(T* a) : shared_ptr<T>(a) {

			}

		public:
			T& val() {
				return *shared_ptr<T>::get();
			}

		public:
			void link(ptr<T>& other) {
				*this = other;
			}
			void unlink() {
				T temp = val();
				shared_ptr<T>::reset(new T());
				val() = temp;
			}

		};
	}
}