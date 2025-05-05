#include "mgil.hpp"

#include <print>
#include <vector>
#include <algorithm>

auto demo1() -> void {
    using namespace mgil;

    using pixel = Pixel<int, rgb_layout_t>;

    // Generate an image view of a gradient image. width = 8, height = 8, initial={0, 0, 0}, x step={1, 2, 0}, y
    // step={0, 0, 3}
    auto gradient_view = gradient(8, 8, pixel{0, 0, 0}, pixel{1, 2, 0}, pixel{0, 0, 3});
    // STL compatibility: just print the image view!
    // Both the std::cout and the newer std::println() are supported.
    // You can also use the ranged-for or old-fashioned iterator loop to manually do the printing
    std::println("{}", gradient_view);

    // Let play with the image view!
    // The MGIL view adaptors can be chained together using the pipe operator "|"
    // Just like the C++ Ranges library!
    // Read adaptor1 | adaptor 2 as adaptor2(adaptor1)
    auto processed_view = gradient_view
                    | crop(2, 2, 4, 4)  // Crop the view at (2, 2) with cropped view width = 4, height = 4
                    | rotate90()        // Rotate the view by 90 degrees clockwise
                    | flipVertical()    // Flip the view vertically
                    | transform([](pixel input) {
                          input.get<red_color_t>() = 0;     // Make red channel all zero
                          input.get<green_color_t>() *= 2;  // Amplify the green channel by factor of 2
                          return input;
                      });

    // Print out the view to see what happened!
    std::println("{}", processed_view);
}

auto demo2() -> void {
    using namespace mgil;

    // MGIL points and pixels all implemented the C++ tuple protocol, so you can
    // retrieve their components in an intuitive way, by using the structured
    // binding

    using pixel = Pixel<int, rgb_layout_t>;

    pixel p{11, 45, 14};
    auto [r, g, b] = p;
    std::println("R: {}, G: {}, B: {}", r, g, b);

    using point = Point<std::ptrdiff_t>;
    point pos{42, 24};
    auto [x, y] = pos;
    std::println("x: {}, y: {}", x, y);

    // You can create image views from C++ ranges easily:
    using gray_pixel = Pixel<int, gray_layout_t>;
    // from either a one-dimensional range with width and height specified...
    std::vector pixels1 = {
        gray_pixel{1}, gray_pixel{2}, gray_pixel{3},
        gray_pixel{4}, gray_pixel{5}, gray_pixel{6},
        gray_pixel{7}, gray_pixel{8}, gray_pixel{9},
    };
    auto from_range_view_1 = fromRange(pixels1, 3, 3);
    std::println("from_range_view_1: {}", from_range_view_1);
    // or a two-dimensional nested range
    std::vector pixels2 = {
        std::vector{gray_pixel{1}, gray_pixel{2}, gray_pixel{3}},
        std::vector{gray_pixel{4}, gray_pixel{5}, gray_pixel{6}},
        std::vector{gray_pixel{7}, gray_pixel{8}, gray_pixel{9}},
    };
    auto from_range_view_2 = fromRange(pixels2);
    std::println("from_range_view_2: {}", from_range_view_2);

    // MGIL views can integrate seamlessly with STL algorithms:
    auto max_pixel = std::ranges::max_element(from_range_view_1);
    std::println("max_pixel: {}", *max_pixel);
}

auto demo3() -> void {
    using namespace mgil;

    // You can read image files with a unified interface readImage().
    // The support for different image format is provided by different
    // image file I/O class.
    auto image = readImage<BMPFileIO<>>("./demo3.bmp");

    // Handle possible exceptions using C++23 std::expected
    if (not image.has_value()) {
        // The error code is provided by a enum class
        std::println("Failed to read image with error code {}",
            std::to_underlying(image.error()));
        return;
    }

    // Now we can play with the image!
    // The readImage will return an owning image container.
    // To use it with the view adaptors, first you should convert it into a view
    // by the .toView() function
    auto width = image.value().toView().width() / 2;
    auto height = image.value().toView().height() / 2;
    auto processed_view_1 = image.value().toView()
        | rotate180()   // rotate the image by 180 degrees clockwise
        | transform([](auto pixel) {
            pixel.template get<red_color_t>() = 0;
            pixel.template get<green_color_t>() = 0;
            // extracts the blue channel by setting other two all zero
            return pixel;
        })
        | nearest(width, height);   // subsample the image by nearest linear interpolation

    auto processed_view_2 = image.value().toView()
        | flipVertical()    // flip the image vertically
        | transform([](auto pixel) {
            pixel.template get<red_color_t>() = 0;
            pixel.template get<blue_color_t>() = 0;
            // extracts the green channel by setting other two all zero
            return pixel;
        })
        | nearest(width, height);   // subsample the image by nearest linear interpolation

    // concat the two image views vertically
    auto final_processed_view = concatVertical(processed_view_1, processed_view_2);

    // then we can write the view to file to see what happened!
    writeImage<BMPFileIO<>>(final_processed_view, "./demo3_result.bmp");
}

auto demo4() -> void {
    using namespace mgil;

    auto image = readImage<BMPFileIO<>>("./demo4.bmp");

    // Handle possible exceptions using C++23 std::expected
    if (not image.has_value()) {
        // The error code is provided by a enum class
        std::println("Failed to read image with error code {}",
            std::to_underlying(image.error()));
        return;
    }

    // Using the Gaussian blur algorithm to blur the image.
    auto blurred_image = gaussianBlur(image.value().toView(), 10.f);

    // then we can write the view to file，
    writeImage<BMPFileIO<>>(blurred_image.toView(), "./demo4_result.bmp");
}

auto demo5() -> void {
    using namespace mgil;
    // Generate an image view using a generator function
    auto view = generate(2, 2, [](auto const x, auto const y) {
        std::println("generate({}, {})", x, y);
        return Pixel<int, gray_layout_t>{x + y};
    });
    // Transform it
    auto transformed_view = view | transform([](auto pixel) {
        std::println("transform({})", pixel);
        return pixel;
    });
    // Traverse it
    for (auto const &pixel : transformed_view) {
        std::println("Loop: {}", pixel);
    }
}

auto main() -> int {
    demo5();
}
