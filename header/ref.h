#pragma once
#include "pch.h"

namespace affix_base {
	namespace data {
		template<class T>
		class ref {
		public:
			T* m_pointer = nullptr;

		public:
			ref() {

			}
			ref(T* a_pointer) {
				m_pointer = a_pointer;
			}

		public:
			void operator=(T a_other) {
				*m_pointer = a_other;
			}
			void operator=(T* a_pointer) {
				m_pointer = a_pointer;
			}
			void operator=(ref<T>& a_other) {
				*m_pointer = *a_other.pointer;
			}
			operator T& () {
				return *m_pointer;
			}

		};
	}
}