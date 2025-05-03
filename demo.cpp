#include "mgil.hpp"

#include <iostream>
#include <print>

auto main() -> int {
    mgil::Pixel<mgil::Double_01, mgil::rgba_layout_t> const pixel(0.1, 0.2, 0.3, 0.4);
    std::cout << pixel << std::endl;
    std::println("{}", pixel);
}