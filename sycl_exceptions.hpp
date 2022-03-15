 /*
 * Copyright (c) 2021-2022, Virginia Tech.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <CL/sycl.hpp>
#include <iostream>
static auto sycl_exception_handler = [](sycl::exception_list exceptions) {
  for (const std::exception_ptr &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception &e) {
      std::cerr << "Unhandled exception occured!\n\t" << e.what() << std::endl;
      exit(e.get_cl_code());
    }
  }
};
