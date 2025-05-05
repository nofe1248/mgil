#pragma once

#ifndef MGIL_H
#define MGIL_H

/*
 * MGIL: Modern Generic Image Library
 *
 * Zhige Chen, Southern University of Science and Technology
 *
 * This library is a project assignment for the CS219 (Advanced Programming) course.
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <compare>
#include <concepts>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <iterator>
#include <mdspan>
#include <numeric>
#include <ostream>
#include <random>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "mgil.hpp"

// A Stripped Version of qlibs/mp, a lightweight, self-contained C++ metaprogramming library
// See https://github.com/qlibs/mp/

namespace mgil::details::mp {
    using size_t = decltype(sizeof(int));

    namespace utility {
        template<class T, T...>
        struct integer_sequence {};
        template<size_t... Ns>
        using index_sequence = integer_sequence<size_t, Ns...>;
        template<size_t N>
        using make_index_sequence =
#if defined(__clang__) || defined(_MSC_VER)
                __make_integer_seq<integer_sequence, size_t, N>;
#else
                index_sequence<__integer_pack(N)...>;
#endif
        template<class T>
        auto declval() -> T &&;
    } // namespace utility

    namespace concepts {
        template<class TRange>
        concept range = requires(TRange range) {
            range.begin();
            range.end();
        };
        template<class Fn, class... TArgs>
        concept invocable = requires(Fn fn, TArgs... args) { fn(args...); };
    } // namespace concepts

    template<class T, size_t N>
    struct array {
        using value_type = T;
        [[nodiscard]] constexpr auto begin() const {
            return &data[0];
        }
        [[nodiscard]] constexpr auto begin() {
            return &data[0];
        }
        [[nodiscard]] constexpr auto end() const {
            return &data[0] + N;
        }
        [[nodiscard]] constexpr auto end() {
            return &data[0] + N;
        }
        [[nodiscard]] constexpr auto operator[](size_t I) const {
            return data[I];
        }
        [[nodiscard]] constexpr auto &operator[](size_t I) {
            return data[I];
        }
        [[nodiscard]] constexpr auto size() const {
            return N;
        }
        [[nodiscard]] constexpr auto operator==(const array &other) const -> bool {
            for (auto i = 0u; i < N; ++i) {
                if (data[i] != other.data[i]) {
                    return false;
                }
            }
            return true;
        }
        [[nodiscard]] constexpr auto operator!=(const array &other) const -> bool {
            return not operator==(other);
        }
        T data[N];
    };
    template<class T, class... Ts>
    array(T, Ts...) -> array<T, 1u + sizeof...(Ts)>;

    template<class T, size_t N = 1024u>
    struct vector {
        using value_type = T;
        constexpr vector() = default;
        constexpr vector(size_t size) : size_{size} {
        }
        constexpr explicit vector(const auto &...ts)
            requires(requires { T(ts); } and ...)
            : data{ts...}, size_{sizeof...(ts)} {
        }
        constexpr vector(concepts::range auto range) {
            for (const auto &t: range) {
                data[size_++] = t;
            }
        }
        constexpr void push_back(const T &t) {
            data[size_++] = t;
        }
        constexpr void emplace_back(T &&t) {
            data[size_++] = static_cast<T &&>(t);
        }
        [[nodiscard]] constexpr auto begin() const {
            return &data[0];
        }
        [[nodiscard]] constexpr auto begin() {
            return &data[0];
        }
        [[nodiscard]] constexpr auto end() const {
            return &data[0] + size_;
        }
        [[nodiscard]] constexpr auto end() {
            return &data[0] + size_;
        }
        [[nodiscard]] constexpr auto operator[](size_t i) const {
            return data[i];
        }
        [[nodiscard]] constexpr auto &operator[](size_t i) {
            return data[i];
        }
        [[nodiscard]] constexpr auto size() const {
            return size_;
        }
        [[nodiscard]] constexpr auto resize(size_t size) {
            size_ = size;
        }
        [[nodiscard]] constexpr auto capacity() const {
            return N;
        }
        [[nodiscard]] constexpr auto operator==(const vector<T> &other) const -> bool {
            if (size_ != other.size_) {
                return false;
            }
            for (auto i = 0u; i < size_; ++i) {
                if (data[i] != other.data[i]) {
                    return false;
                }
            }
            return true;
        }
        template<size_t size>
        [[nodiscard]] constexpr auto operator!=(const vector<T> &other) const -> bool {
            return not operator==(other);
        }
        constexpr void clear() {
            size_ = {};
        }

        union {
            T data[N]{};
        }; // __cpp_trivial_union
        size_t size_{};
    };
    template<class T, class... Ts>
    vector(T, Ts...) -> vector<T>;

    enum class info : size_t {
    };

    namespace detail {
        template<info>
        struct info {
            constexpr auto friend get(info);
        };
        template<class T>
        struct meta {
            using value_type = T;
            template<size_t left = 0u, size_t right = 1024u - 1u>
            static constexpr auto gen() -> size_t {
                if constexpr (left >= right) {
                    return left + requires { get(info<mp::info{left}>{}); };
                } else if constexpr (constexpr auto mid = left + (right - left) / 2u;
                                     requires { get(info<mp::info{mid}>{}); }) {
                    return gen<mid + 1u, right>();
                } else {
                    return gen<left, mid - 1u>();
                }
            }
            static constexpr auto id = mp::info{gen()};
            constexpr auto friend get(info<id>) {
                return meta{};
            }
        };
        void failed();
    } // namespace detail

    template<class T>
    inline constexpr info meta = detail::meta<T>::id;

    template<info meta>
    using type_of = typename decltype(get(detail::info<meta>{}))::value_type;

    template<template<class...> class T>
    [[nodiscard]] constexpr auto apply(concepts::invocable auto expr) {
        constexpr concepts::range auto range = expr();
        return [range]<size_t... Ns>(utility::index_sequence<Ns...>) {
            return T<type_of<range[Ns]>...>{};
        }(utility::make_index_sequence<range.size()>{});
    }

    template<template<class...> class T, concepts::range auto range>
    inline constexpr auto apply_v = []<size_t... Ns>(utility::index_sequence<Ns...>) {
        return T<type_of<range[Ns]>...>{};
    }(utility::make_index_sequence<range.size()>{});

    template<template<class...> class T, concepts::range auto range>
    using apply_t = decltype([]<size_t... Ns>(utility::index_sequence<Ns...>) {
        return utility::declval<T<type_of<range[Ns]>...>>();
    }(utility::make_index_sequence<range.size()>{}));

    template<class Fn, class T = decltype([] {})>
    [[nodiscard]] inline constexpr auto invoke(Fn &&fn, info meta) {
        constexpr auto dispatch = [&]<size_t... Ns>(utility::index_sequence<Ns...>) {
            return array {
                []<info N> {
                    return +[](Fn fn) {
                        if constexpr (requires { fn.template operator()<N>(); }) {
                            return fn.template operator()<N>();
                        }
                    };
                }.template operator()<info{Ns}>()...
            };
        }(utility::make_index_sequence<size_t(mp::meta<T>)>{});
        return dispatch[size_t(meta)](fn);
    }

    template<template<class...> class T, class... Ts, auto = [] {}>
    [[nodiscard]] inline constexpr auto invoke(info meta) {
        return invoke(
                []<info meta> {
                    using type = type_of<meta>;
                    if constexpr (requires { T<Ts..., type>::value; }) {
                        return T<Ts..., type>::value;
                    } else {
                        return mp::meta<typename T<Ts..., type>::type>;
                    }
                },
                meta);
    }

    template<size_t N>
    inline constexpr auto unroll = [](auto &&fn) {
        const auto invoke = [&]<size_t I> {
            if constexpr (requires { fn.template operator()<size_t{}>(); }) {
                fn.template operator()<I>();
            } else {
                fn();
            }
        };
        [&]<size_t... Ns>(utility::index_sequence<Ns...>) {
            (invoke.template operator()<Ns>(), ...);
        }(utility::make_index_sequence<N>{});
    };

    template<concepts::range auto range>
    inline constexpr auto for_each = [](auto &&fn) {
        [fn]<size_t... Ns>(utility::index_sequence<Ns...>) {
            (fn.template operator()<range[Ns]>(), ...);
        }(utility::make_index_sequence<range.size()>{});
    };
} // namespace mgil::details::mp

#if __cpp_deleted_function >= 202403L
#define DELETE_DEF(str) delete (str)
#elif
#define DELETE_DEF(str) delete
#endif

// Metaprogramming Utils
// WARNING: TONS of template tricks

namespace mgil::details {
    // A simple utility to extract nth element from a pack
#if __cpp_pack_indexing >= 202311L
    template<std::size_t Index, typename... Types>
    struct nth_type {
        using type = Types...[Index];
    };
    template<std::size_t Index, typename... Types>
    using nth_type_t = typename nth_type<Index, Types...>::type;

    template<std::size_t Index, auto... Values>
    struct nth_nttp {
        static constexpr auto value = Values...[Index];
    };
    template<std::size_t Index, auto... Values>
    constexpr auto nth_nttp_v = nth_nttp<Index, Values...>::value;
#elif
    template<std::size_t Index, typename Head, typename... Types>
    struct nth_type {
        using type = typename nth_type_impl<Index - 1, Pack...>::type;
    };
    template<typename Head, typename... Types>
    struct nth_type<0, Head, Types...> {
        using type = Head;
    };
    template<std::size_t Index, typename Head, typename... Types>
    using nth_type_t = nth_type<Index, Head, Pack...>::type;

    template<std::size_t Index, typename Head, auto... Values>
    struct nth_nttp {
        static constexpr auto value = typename nth_type_impl<Index - 1, Values...>::value;
    };
    template<typename Head, auto... Values>
    struct nth_nttp<0, Head, Values...> {
        static constexpr auto value = Head;
    };
    template<std::size_t Index, typename Head, auto... Values>
    constexpr auto nth_nttp_v = nth_type<Index, Head, Values...>::value;
#endif

    // A not-so-simple type list
    template<typename... Types>
    struct type_list {
        static constexpr size_t size = sizeof...(Types);
        template<std::size_t Index>
        using NthType = typename nth_type<Index, Types...>::type;
        static constexpr auto getVector() noexcept {
            return std::vector<mp::info>{mp::meta<Types>...};
        }
        static constexpr auto getMPVector() noexcept {
            return mp::vector<mp::info>{mp::meta<Types>...};
        }
        template<template<typename> class Pred>
        static constexpr auto allConformTo() noexcept -> bool {
            return (Pred<Types>::value && ...);
        }
        template<auto Pred>
        static constexpr auto allConformTo() noexcept -> bool {
            return (Pred.template operator()<Types>() && ...);
        }
    };

    // Again a not-so-simple index list
    template<std::size_t... indexes>
    struct index_list {
        static constexpr std::size_t size = sizeof...(indexes);
        template<std::size_t N>
        static constexpr auto get() -> std::size_t {
            return nth_nttp_v<N, indexes...>;
        }
        static constexpr auto getVector() noexcept {
            return std::vector<std::size_t>{indexes...};
        }
        static constexpr auto getMPVector() noexcept {
            return mp::vector<std::size_t>{indexes...};
        }
    };

    template<typename T, template<auto...> class Templ>
    struct is_specialization_of_nttp : std::false_type {};

    template<template<auto...> class Templ, auto... Args>
    struct is_specialization_of_nttp<Templ<Args...>, Templ> : std::true_type {};

    template<typename T, template<auto...> class Templ>
    constexpr bool is_specialization_of_nttp_v = is_specialization_of_nttp<T, Templ>::value;

    template<typename T, template<typename...> class Templ>
    struct is_specialization_of : std::false_type {};

    template<template<typename...> class Templ, typename... Args>
    struct is_specialization_of<Templ<Args...>, Templ> : std::true_type {};

    template<typename T, template<typename...> class Templ>
    constexpr bool is_specialization_of_v = is_specialization_of<T, Templ>::value;

#ifndef MGIL_NO_COMPILE_TIME_TESTING
    static_assert(is_specialization_of_nttp_v<index_list<1, 2, 3>, index_list>);
    static_assert(is_specialization_of_v<std::vector<int>, std::vector>);
#endif

    // Compile-time algorithm: Sort types
    template<typename... Types>
    struct sort_types {
        using type = mp::apply_t<type_list, [] {
            auto vec = std::vector<mp::info>{mp::meta<Types>...};
            std::ranges::sort(vec);
            return vec | std::ranges::to<mp::vector<mp::info>>();
        }()>;
    };

    template<typename... Types>
    struct sort_types<type_list<Types...>> : sort_types<Types...> {};

    // Compile-time algorithm: Deduplicate types
    template<typename... Types>
    struct deduplicate_types {
        using type = mp::apply_t<type_list, [] {
            auto vec = mp::vector{mp::meta<Types>...} |
                       std::views::transform([](auto m) { return mp::invoke<std::remove_cvref>(m); }) |
                       std::views::transform([](auto m) { return mp::invoke<std::remove_all_extents>(m); }) |
                       std::views::transform([](auto m) { return mp::invoke<std::remove_pointer>(m); }) |
                       std::views::transform([](auto m) { return mp::invoke<std::decay>(m); }) |
                       std::ranges::to<std::vector<mp::info>>();
            std::ranges::sort(vec);
            auto ret = std::ranges::unique(vec);
            vec.erase(ret.begin(), ret.end());
            return vec | std::ranges::to<mp::vector<mp::info>>();
        }()>;
    };

    template<typename... Types>
    struct deduplicate_types<type_list<Types...>> : deduplicate_types<Types...> {};

    // Compile-time algorithm: Find a type in a pack of types
    template<typename TypeToFind, typename... Types>
    struct find_type {
        static constexpr std::size_t value = [] {
            auto vec = std::vector<mp::info>{mp::meta<Types>...};
            return std::distance(vec.begin(), std::ranges::find(vec, mp::meta<TypeToFind>));
        }();
    };

    template<typename TypeToFind, typename... Types>
    struct find_type<TypeToFind, type_list<Types...>> : find_type<TypeToFind, Types...> {};

    template<typename TypeToFind, typename... Types>
    constexpr std::size_t find_type_v = find_type<TypeToFind, Types...>::value;

    // Compile-time algorithm: Find if a type in a pack of types
    template<typename TypeToFind, typename... Types>
    struct contain_type {
        static constexpr bool value = [] {
            auto vec = std::vector<mp::info>{mp::meta<Types>...};
            return std::ranges::contains(vec, mp::meta<TypeToFind>);
        }();
    };

    template<typename TypeToFind, typename... Types>
    struct contain_type<TypeToFind, type_list<Types...>> : contain_type<TypeToFind, Types...> {};

    template<typename TypeToFind, typename... Types>
    constexpr bool contain_type_v = contain_type<TypeToFind, Types...>::value;

    template<typename TypeList, typename IndexList>
        requires is_specialization_of_v<TypeList, type_list> and is_specialization_of_nttp_v<IndexList, index_list> and
                 (TypeList::size == IndexList::size)
    struct rearrange_type_list {
        using type = mp::apply_t<type_list, [] {
            auto const type_vec = TypeList::getVector();
            auto result_vec = std::vector<mp::info>{};
            for (auto const index_vec = IndexList::getVector(); auto const index: index_vec) {
                result_vec.push_back(type_vec.at(index));
            }
            return result_vec | std::ranges::to<mp::vector<mp::info>>();
        }()>;
    };

    template<typename TypeList, typename IndexList>
    using rearrange_type_list_t = typename rearrange_type_list<TypeList, IndexList>::type;

#ifndef MGIL_NO_COMPILE_TIME_TESTING
    static_assert(std::same_as<type_list<int, float, char>,
                               rearrange_type_list_t<type_list<float, char, int>, index_list<2, 0, 1>>>);
#endif

    // The base class of every image adaptor closure
    struct image_adaptor_closure_tag {};

    inline namespace concepts {
        template<typename Lhs, typename Rhs, typename Image>
        concept PipeInvocable = requires { std::declval<Rhs>()(std::declval<Lhs>()(std::declval<Image>())); };

        template<typename Type>
        concept IsImageAdaptorClosure = std::derived_from<Type, image_adaptor_closure_tag>;

        template<typename Adaptor, typename... Args>
        concept AdaptorInvocable = requires(Adaptor adaptor) { adaptor(std::declval<Args>()...); };
    } // namespace concepts

    template<typename T, typename U>
    using like_t = decltype(std::forward_like<T>(std::declval<U>()));

    // Type that represents the pipe structure
    template<typename Lhs, typename Rhs>
    struct Pipe : image_adaptor_closure_tag {
        [[no_unique_address]] Lhs _lhs;
        [[no_unique_address]] Rhs _rhs;

        template<typename Tp, typename Up>
        constexpr Pipe(Tp &&lhs, Up &&rhs) : _lhs(std::forward<Tp>(lhs)), _rhs(std::forward<Up>(rhs)) {
        }

        // Invoke _rhs(_lhs(img)) according to the value category of this image adaptor closure object
#if __cpp_explicit_this_parameter >= 202110L
        template<typename Self, typename Image>
            requires PipeInvocable<like_t<Self, Lhs>, like_t<Self, Rhs>, Image>
        constexpr auto operator()(this Self &&self, Image &&img) {
            return std::forward<Self>(self)._rhs(std::forward<Self>(self)._lhs(std::forward<Image>(img)));
        }
#elif
        template<typename Image>
            requires PipeInvocable<Lhs const &, Rhs const &, Image>
        constexpr auto operator()(Image &&img) const & {
            return _rhs(_lhs(std::forward<Image>(img)));
        }

        template<typename Image>
            requires PipeInvocable<Lhs &&, Rhs &&, Image>
        constexpr auto operator()(Image &&img) && {
            return std::move(_rhs)(std::move(_lhs)(std::forward<Image>(img)));
        }

        template<typename Image>
        constexpr auto operator()(Image &&img) const && =
                DELETE_DEF("Invoking a pipe object with const r-value references is prohibited");
#endif
    };

    template<typename Fn>
    struct pipeable : Fn, image_adaptor_closure_tag {
        constexpr explicit pipeable(Fn &&f) : Fn(std::move(f)) {
        }
    };

    // img | adaptor is equivalent to adaptor(img)
    template<typename Self, typename Image>
        requires IsImageAdaptorClosure<Self> and AdaptorInvocable<Self, Image>
    constexpr auto operator|(Image &&img, Self &&self) {
        return std::forward<Self>(self)(std::forward<Image>(img));
    }

    // adaptor1 | adaptor2 is equivalent to adaptor2(adaptor1)
    template<typename Lhs, typename Rhs>
        requires IsImageAdaptorClosure<Lhs> and IsImageAdaptorClosure<Rhs>
    constexpr auto operator|(Lhs &&lhs, Rhs &&rhs) {
        return Pipe<std::decay_t<Lhs>, std::decay_t<Rhs>>{std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)};
    }

    template<typename T>
    concept IsArithmetic = std::is_arithmetic_v<T>;

    template<typename OutPtr, typename In>
    constexpr auto ptr_reinterpret_cast(In *ptr) -> OutPtr {
        return static_cast<OutPtr>(static_cast<void *>(ptr));
    }

    template<typename OutPtr, typename In>
    constexpr auto ptr_reinterpret_cast_const(In *ptr) -> OutPtr {
        return static_cast<OutPtr>(static_cast<void const *>(ptr));
    }

    // Helper classes and concepts
    template<class Reference>
    struct arrow_proxy {
        Reference r;
        auto operator->() -> Reference * {
            return &r;
        }
    };

    template<typename T>
    concept ImplsDistanceTo = requires(T const it) { it.distance_to(std::declval<T>()); };

    template<typename>
    struct infer_difference_type {
        using type = std::ptrdiff_t;
    };
    template<typename T>
        requires ImplsDistanceTo<T>
    struct infer_difference_type<T> {
        static const T &it_;
        using type = decltype(it_.distance_to(it_));
    };
    template<typename T>
    using infer_difference_type_t = typename infer_difference_type<T>::type;

    template<typename T>
    struct infer_value_type {
        static const T &it_;
        using type = std::remove_cvref_t<decltype(*it_)>;
    };
    template<typename T>
        requires requires { typename T::value_type; }
    struct infer_value_type<T> {
        using type = typename T::value_type;
    };
    template<typename T>
    using infer_value_type_t = typename infer_value_type<T>::type;

    template<typename T>
    concept ImplsIncrement = requires(T t) { t.increment(); };
    template<typename T>
    concept ImplsDecrement = requires(T t) { t.decrement(); };
    template<typename T>
    concept ImplsAdvance = requires(T it, infer_difference_type_t<T> const offset) { it.advance(offset); };
    template<typename T>
    concept ImplsEqualTo = requires(T const it) {
        { it.equal_to(it) } -> std::same_as<bool>;
    };
    template<typename T>
    concept IsRandomAccessIterator = ImplsAdvance<T> and ImplsDistanceTo<T>;
    template<typename T>
    concept IsBidirectionalIterator = IsRandomAccessIterator<T> and ImplsDecrement<T>;
    template<typename T>
    concept IsSinglePassIterator = static_cast<bool>(T::single_pass_iterator);
    template<typename Arg, typename It>
    concept DifferenceTypeArg = std::convertible_to<Arg, infer_difference_type_t<It>>;

    // A modernized version of boost::iterator_facade
    // Helps to reduce the boilerplate code when writing iterators
    template<typename Derived>
    class iterator_facade {
    public:
        using self_type = Derived;

    private:
        auto self() -> self_type & {
            return static_cast<self_type &>(*this);
        }
        auto self() const -> self_type const & {
            return static_cast<self_type const &>(*this);
        }

    public:
        decltype(auto) operator*() const {
            return self().dereference();
        }
        auto operator->() const {
            decltype(auto) ref = **this;
            if constexpr (std::is_reference_v<decltype(ref)>) {
                return std::addressof(ref);
            } else {
                return arrow_proxy(std::move(ref));
            }
        }

        friend auto operator==(self_type const &lhs, self_type const &rhs) -> bool {
            return lhs.equal_to(rhs);
        }

        auto operator++() -> self_type & {
            if constexpr (ImplsIncrement<self_type>) {
                self().increment();
            } else {
                static_assert(ImplsAdvance<self_type>,
                              "Iterator subclass must provide either .advance() or .increment() functions");
                self().advance(1);
            }
            return self();
        }
        auto operator++(int) -> self_type {
            auto copy = self();
            ++*this;
            return copy;
        }
        auto operator--() -> self_type & {
            if constexpr (ImplsDecrement<self_type>) {
                self().decrement();
            } else {
                static_assert(ImplsAdvance<self_type>,
                              "Iterator subclass must provide either .advance() or .decrement() functions");
                self().advance(-1);
            }
            return self();
        }
        auto operator--(int) -> self_type
            requires ImplsDecrement<self_type>
        {
            auto copy = self();
            --*this;
            return copy;
        }

        friend auto operator+=(self_type &self, DifferenceTypeArg<self_type> auto offset) -> self_type &
            requires ImplsAdvance<self_type>
        {
            self.advance(offset);
            return self;
        }
        friend auto operator+(self_type self, DifferenceTypeArg<self_type> auto offset) -> self_type
            requires ImplsAdvance<self_type>
        {
            return self += offset;
        }
        friend auto operator+(DifferenceTypeArg<self_type> auto offset, self_type self) -> self_type
            requires ImplsAdvance<self_type>
        {
            return self += offset;
        }
        friend auto operator-(self_type self, DifferenceTypeArg<self_type> auto offset) -> self_type
            requires ImplsAdvance<self_type>
        {
            return self + -offset;
        }
        friend auto operator-=(self_type &self, DifferenceTypeArg<self_type> auto offset) -> self_type &
            requires ImplsAdvance<self_type>
        {
            return self = self - offset;
        }
        decltype(auto) operator[](DifferenceTypeArg<self_type> auto offset)
            requires ImplsAdvance<self_type>
        {
            return *(self() + offset);
        }
        decltype(auto) operator[](DifferenceTypeArg<self_type> auto offset) const
            requires ImplsAdvance<self_type>
        {
            return *(self() + offset);
        }
        friend auto operator-(self_type const &lhs, self_type const &rhs)
            requires ImplsDistanceTo<self_type>
        {
            return rhs.distance_to(lhs);
        }
        friend auto operator<=>(self_type const &lhs, self_type const &rhs)
            requires ImplsDistanceTo<self_type>
        {
            return (lhs - rhs) <=> 0;
        }
    };
} // namespace mgil::details

template<typename It>
    requires std::derived_from<It, mgil::details::iterator_facade<It>>
struct std::iterator_traits<It> {
    static const It &it_;
    using reference = decltype(*it_);
    using pointer = decltype(it_.operator->());
    using difference_type = mgil::details::infer_difference_type_t<It>;
    using value_type = mgil::details::infer_value_type_t<It>;

    using iterator_category =
            conditional_t<mgil::details::IsRandomAccessIterator<It>, random_access_iterator_tag,
                          conditional_t<mgil::details::IsBidirectionalIterator<It>, bidirectional_iterator_tag,
                                        conditional_t<mgil::details::IsSinglePassIterator<It>, input_iterator_tag,
                                                      forward_iterator_tag>>>;
    using iterator_concept = iterator_category;
};

namespace mgil::details {
    template<typename Loc, typename XIt, typename YIt, typename PointType>
        requires std::same_as<typename PointType::value_type, typename std::iterator_traits<XIt>::difference_type>
    class position_locator_base {
    public:
        using x_iterator = XIt;
        using y_iterator = YIt;

        using value_type = typename std::iterator_traits<x_iterator>::value_type;
        using reference = typename std::iterator_traits<x_iterator>::reference;
        using coordinate_type = typename std::iterator_traits<x_iterator>::difference_type;
        using difference_type = PointType;
        using point_type = PointType;
        using x_coordinate_type = coordinate_type;
        using y_coordinate_type = coordinate_type;

    private:
        constexpr auto self() -> Loc & {
            return static_cast<Loc &>(*this);
        }
        constexpr auto self() const -> Loc const & {
            return static_cast<Loc const &>(*this);
        }

    public:
        constexpr auto operator==(Loc const &that) const -> bool {
            return self() == that;
        }
        constexpr auto operator!=(Loc const &that) const -> bool {
            return not(self() == that);
        }

        constexpr auto xAt(x_coordinate_type dx, y_coordinate_type dy) const -> x_iterator {
            Loc temp = self();
            temp += point_type(dx, dy);
            return temp.x();
        }
        constexpr auto xAt(difference_type d) const -> x_iterator {
            Loc temp = self();
            temp += d;
            return temp.x();
        }
        constexpr auto yAt(x_coordinate_type dx, y_coordinate_type dy) const -> x_iterator {
            Loc temp = self();
            temp += point_type(dx, dy);
            return temp.y();
        }
        constexpr auto yAt(difference_type d) const -> x_iterator {
            Loc temp = self();
            temp += d;
            return temp.y();
        }
        constexpr auto xyAt(x_coordinate_type dx, y_coordinate_type dy) const -> Loc {
            Loc temp = self();
            temp += point_type(dx, dy);
            return temp;
        }
        constexpr auto xyAt(difference_type d) const -> Loc {
            Loc temp = self();
            temp += d;
            return temp;
        }

        constexpr auto operator()(x_coordinate_type dx, y_coordinate_type dy) const -> reference {
            return *xAt(dx, dy);
        }
        constexpr auto operator[](difference_type const &d) const -> reference {
            return *xAt(d.x(), d.y());
        }

        constexpr auto operator*() const -> reference {
            return *self().x();
        }

        constexpr auto operator+=(difference_type const &d) -> Loc & {
            self().pos().x() += d.x();
            self().pos().y() += d.y();
            return self();
        }
        constexpr auto operator-=(difference_type const &d) const -> Loc & {
            self().pos().x() -= d.x();
            self().pos().y() -= d.y();
            return self();
        }

        constexpr auto operator+(difference_type const &d) const -> Loc {
            return xyAt(d);
        }
        constexpr auto operator-(difference_type const &d) const -> Loc {
            return xyAt(-d);
        }

        using cached_location_type = difference_type;
        constexpr auto cacheLocation(difference_type const &d) const -> cached_location_type {
            return d;
        }
        constexpr auto cacheLocation(x_coordinate_type x, y_coordinate_type y) const -> cached_location_type {
            return difference_type(x, y);
        }

        template<typename Deref, bool IsTransposed, typename PointType_>
        friend class position_locator;
    };
} // namespace mgil::details

// Traits
namespace mgil::inline traits {
    template<typename T>
    struct ChannelTraitsImpl {};

    template<typename T>
        requires std::integral<T> or std::floating_point<T>
    struct ChannelTraitsImpl<T> {
        using value_type = T;
        using reference = value_type &;
        using pointer = value_type *;
        using const_reference = value_type const &;
        using const_pointer = value_type const *;
        static constexpr bool is_mutable = true;
        static constexpr auto minValue() -> value_type {
            return std::numeric_limits<value_type>::min();
        }
        static constexpr auto maxValue() -> value_type {
            return std::numeric_limits<value_type>::max();
        }
        static constexpr auto setValue(T &t, value_type v) -> void {
            t = v;
        }
        static constexpr auto getValue(T const &t) -> T {
            return t;
        }
    };

    template<typename T>
        requires std::integral<T> or std::floating_point<T>
    struct ChannelTraitsImpl<T const> : ChannelTraitsImpl<T> {
        using reference = T const &;
        using pointer = T const *;
        static constexpr bool is_mutable = false;
    };

    struct channel_tag {};

    template<typename T>
        requires std::is_class_v<T> and std::derived_from<T, channel_tag>
    struct ChannelTraitsImpl<T> {
        using value_type = typename T::value_type;
        using reference = typename T::reference;
        using pointer = typename T::pointer;
        using const_reference = typename T::const_reference;
        using const_pointer = typename T::const_pointer;
        static constexpr bool is_mutable = T::is_mutable;
        static constexpr auto minValue() -> value_type {
            return T::minValue();
        }
        static constexpr auto maxValue() -> value_type {
            return T::maxValue();
        }
        static constexpr auto setValue(T &t, value_type v) -> void {
            t.setValue(v);
        }
        static constexpr auto getValue(T const &t) -> value_type {
            return t.getValue();
        }
    };

    template<typename T>
    struct ChannelTraits : ChannelTraitsImpl<T> {};

    template<typename T>
    struct ChannelTraits<T &> : ChannelTraits<T> {};

    template<typename T>
    struct ChannelTraits<T const &> : ChannelTraits<T> {};

    struct pixel_iterator_tag {};

    template<typename T>
    struct PixelIteratorTraits {};

    template<typename T>
        requires std::derived_from<T, pixel_iterator_tag>
    struct PixelIteratorTraits<T> {
        using const_iterator = typename T::const_iterator;
        using size_type = typename T::size_type;
        using x_coordinate_type = typename T::x_coordinate_type;
        using y_coordinate_type = typename T::y_coordinate_type;
        static constexpr bool is_mutable = T::is_mutable;
    };

    template<typename T>
    struct PixelTraits;
} // namespace mgil::inline traits

// Concepts for pixels, images, etc.
namespace mgil::inline concepts {
    // A point defined the location of a pixel inside an image.
    template<typename T>
    concept IsPoint = std::regular<T> && requires(T t, T const ct, T u, int i) {
        typename T::value_type;
        { t.x() } -> std::same_as<typename T::value_type &>;
        { t.y() } -> std::same_as<typename T::value_type &>;
        { ct.x() } -> std::same_as<typename T::value_type const &>;
        { ct.y() } -> std::same_as<typename T::value_type const &>;
        { t + u } -> std::same_as<T>;
        { t - u } -> std::same_as<T>;
        { t *u } -> std::same_as<T>;
        { t / u } -> std::same_as<T>;
        { t + i } -> std::same_as<T>;
        { t - i } -> std::same_as<T>;
        { t *i } -> std::same_as<T>;
        { t / i } -> std::same_as<T>;
    };

    // A channel indicates the intensity of a color component
    template<typename T>
    concept IsChannel = std::regular<T> and requires(T t, typename ChannelTraits<T>::value_type v) {
        typename ChannelTraits<T>::value_type;
        typename ChannelTraits<T>::reference;
        typename ChannelTraits<T>::pointer;
        typename ChannelTraits<T>::const_reference;
        typename ChannelTraits<T>::const_pointer;
        { ChannelTraits<T>::is_mutable } -> std::convertible_to<bool>;
        { ChannelTraits<T>::minValue() } -> std::convertible_to<typename ChannelTraits<T>::value_type>;
        { ChannelTraits<T>::maxValue() } -> std::convertible_to<typename ChannelTraits<T>::value_type>;
        { ChannelTraits<T>::setValue(t, v) };
        { ChannelTraits<T>::getValue(t) } -> std::convertible_to<typename ChannelTraits<T>::value_type>;
    };

#ifndef MGIL_NO_COMPILE_TIME_TESTING
    static_assert(IsChannel<int>);
    static_assert(IsChannel<float>);
    static_assert(IsChannel<double>);
#endif

    template<typename T, typename U>
    concept IsChannelsCompatible =
            std::same_as<typename ChannelTraits<T>::value_type, typename ChannelTraits<U>::value_type> and
            IsChannel<T> and IsChannel<U>;

    template<typename Src, typename Dst>
    concept IsChannelsConvertible = IsChannel<Src> and IsChannel<Dst> and requires(Src src) {
        { ChannelTraits<Src>::template convertTo<Dst>(src) } -> std::same_as<Dst>;
    };

    // Color Space and Layout concepts
    template<typename T>
    concept IsColorSpace = details::is_specialization_of_v<T, details::type_list> and (T::size >= 1 and T::size <= 4);

    struct color_space_component_tag {};

    template<typename T>
    concept IsColorSpaceComponent = std::derived_from<T, color_space_component_tag>;

    template<typename ColorSpace_, typename ChannelMapping_>
        requires details::is_specialization_of_nttp_v<ChannelMapping_, details::index_list> and
                 details::is_specialization_of_v<ColorSpace_, details::type_list> and
                 (ColorSpace_::template allConformTo<[]<typename T>() { return IsColorSpaceComponent<T>; }>()) and
                 (ColorSpace_::size == ChannelMapping_::size)
    struct ColorSpaceLayout {
        using color_space = ColorSpace_;
        using channel_mapping = ChannelMapping_;
        using mapped_color_space = details::rearrange_type_list_t<ColorSpace_, ChannelMapping_>;
    };

    template<typename T>
    concept IsLayout = details::is_specialization_of_v<T, ColorSpaceLayout>;

    template<typename T, typename U>
    concept IsLayoutCompatible =
            IsLayout<T> and IsLayout<U> and std::same_as<typename T::color_space, typename U::color_space>;

    // A pixel is a set of channels defining the color at a given point in an image
    template<typename T>
    concept IsPixel = requires(T t, T u, T const ct, std::size_t i) {
        typename T::layout_type;
        typename T::channel_type;
        typename T::value_type;
        typename T::reference;
        typename T::const_reference;

        requires IsLayout<typename T::layout_type>;
        requires IsChannel<typename T::channel_type>;

        { t.get(i) } -> std::convertible_to<typename T::channel_type const &>;
        { t.template getSemantic<0>() } -> std::convertible_to<typename T::channel_type const &>;
        { t.set(i, std::declval<typename T::channel_type>()) };
        { t[i] } -> std::convertible_to<typename T::channel_type &>;
        { ct[i] } -> std::convertible_to<typename T::channel_type const &>;
        { t + u } -> std::convertible_to<T>;
        { t - u } -> std::convertible_to<T>;
        { t *u } -> std::convertible_to<T>;
        { t / u } -> std::convertible_to<T>;
        { t + std::declval<typename T::channel_type>() } -> std::convertible_to<T>;
        { t - std::declval<typename T::channel_type>() } -> std::convertible_to<T>;
        { t *std::declval<typename T::channel_type>() } -> std::convertible_to<T>;
        { t / std::declval<typename T::channel_type>() } -> std::convertible_to<T>;
    } and std::equality_comparable<T>;

    template<typename T, typename U>
    concept IsPixelsCompatible = IsChannelsCompatible<typename T::channel_type, typename U::channel_type> and
                                 IsLayoutCompatible<typename T::layout_type, typename U::layout_type>;

    template<typename T, typename U>
    concept IsPixelsConvertible = (IsChannelsConvertible<typename T::channel_type, typename U::channel_type> or
                                   IsChannelsCompatible<typename T::channel_type, typename U::channel_type>) and
                                  IsLayoutCompatible<typename T::layout_type, typename U::layout_type>;

    template<typename T, typename U>
    concept IsPixelsColorConvertible = requires(T src, U dst) {
        { PixelTraits<T>::template convertTo<U>(src) } -> std::convertible_to<U>;
    };

    // Iterator concepts
    // Models a random-access iterator over pixels
    template<typename It>
    concept IsPixelIterator =
            std::random_access_iterator<It> and IsPixel<std::iter_value_t<It>> and requires(It it, It const cit) {
                typename PixelIteratorTraits<It>::const_iterator;
                typename PixelIteratorTraits<It>::size_type;
                typename PixelIteratorTraits<It>::x_coordinate_type;
                typename PixelIteratorTraits<It>::y_coordinate_type;
                { PixelIteratorTraits<It>::is_mutable } -> std::convertible_to<bool>;
                { it.width() } -> std::convertible_to<typename PixelIteratorTraits<It>::x_coordinate_type &>;
                { it.height() } -> std::convertible_to<typename PixelIteratorTraits<It>::y_coordinate_type &>;
                { cit.width() } -> std::convertible_to<typename PixelIteratorTraits<It>::x_coordinate_type const &>;
                { cit.height() } -> std::convertible_to<typename PixelIteratorTraits<It>::y_coordinate_type const &>;
            };

    template<typename Pixel>
        requires IsPixel<Pixel>
    struct identity_deref_adaptor {
        using value_type = Pixel;
        using const_adaptor = identity_deref_adaptor;
    };
    // Encapsulates on-th-fly pixel transformation
    template<typename T>
    concept IsPixelDereferenceAdaptor = std::default_initializable<T> and requires(T t) {
        typename T::const_adaptor;
        typename T::value_type;
        typename T::result_type;
        typename T::reference;
        typename T::const_reference;

        requires IsPixel<typename T::value_type>;
        requires IsPixel<std::remove_cvref_t<typename T::reference>>;
        requires IsPixel<std::remove_cvref_t<typename T::const_reference>>;

        { T::is_mutable } -> std::convertible_to<bool>;
    } or details::is_specialization_of_v<T, identity_deref_adaptor>;
    template<typename ConstT, typename Value, typename Reference, typename ConstReference, typename ResultType,
             bool IsMutable>
    struct deref_base {
        using result_type = ResultType;
        using const_adaptor = ConstT;
        using value_type = Value;
        using reference = Reference;
        using const_reference = ConstReference;
        static constexpr bool is_mutable = IsMutable;
    };
    template<typename D1, typename D2>
        requires IsPixelDereferenceAdaptor<D1> and IsPixelDereferenceAdaptor<D2>
    class deref_compose
        : public deref_base<deref_compose<typename D1::const_adaptor, typename D2::const_adaptor>,
                            typename D1::value_type, typename D1::reference, typename D1::const_reference,
                            typename D1::result_type, D1::is_mutable and D2::is_mutable> {
    public:
        D1 fn1_;
        D2 fn2_;

        using result_type = typename D1::result_type;

        deref_compose() = default;
        deref_compose(D1 const &d1, D2 const &d2) : fn1_(d1), fn2_(d2) {
        }
        deref_compose(deref_compose const &that) : fn1_(that.fn1_), fn2_(that.fn2_) {
        }

        template<typename D1_, typename D2_>
        explicit deref_compose(deref_compose<D1_, D2_> const &that) : fn1_(that.fn1_), fn2_(that.fn2_) {
        }

        auto operator()(auto x) const {
            return fn1_(fn2_(x));
        }
        auto operator()(auto x) {
            return fn1_(fn2_(x));
        }
    };

    // Allow changing the stride between elements (e.g. column iterators)
    template<typename It>
    concept IsStepIterator = std::forward_iterator<It> and IsPixel<std::iter_value_t<It>> and requires(It it) {
        { it.setStep(std::declval<int>()) };
    };
    template<typename It>
    concept IsMutableStepIterator = IsStepIterator<It> and IsPixel<std::iter_value_t<It>> and
                                    std::indirectly_writable<It, std::iter_value_t<It>>;

    // Pixel iterators that advance in bits
    template<typename It>
    concept IsMemoryBasedIterator = std::random_access_iterator<It> and IsPixel<std::iter_value_t<It>> and
                                    requires(It it, std::ptrdiff_t diff) {
                                        { it.bitsStep() } -> std::same_as<std::ptrdiff_t>;
                                        { it.distance(it) } -> std::same_as<std::ptrdiff_t>;
                                        { it.advance(diff) } -> std::same_as<void>;
                                        { it.advanced(diff) } -> std::same_as<It>;
                                        { it.advancedRef(diff) } -> std::same_as<typename It::reference>;
                                    };

    template<typename It>
    concept IsPixelIteratorHasTransposedType = IsPixelIterator<It> and requires { typename It::transposed_type; };

    // Pixel locator concepts
    template<typename Loc>
    concept Is2DLocator = requires(Loc loc, Loc loc2, Loc const cloc) {
        typename Loc::value_type;
        typename Loc::reference;
        typename Loc::difference_type;
        typename Loc::const_locator;
        typename Loc::cached_location_type;
        typename Loc::point_type;
        typename Loc::size_type;

        typename Loc::x_iterator;
        typename Loc::y_iterator;
        typename Loc::x_coordinate_type;
        typename Loc::y_coordinate_type;

        { Loc::is_mutable } -> std::convertible_to<bool>;

        { loc += std::declval<typename Loc::difference_type>() } -> std::convertible_to<Loc &>;
        { loc -= std::declval<typename Loc::difference_type>() } -> std::convertible_to<Loc &>;
        { loc + std::declval<typename Loc::difference_type>() } -> std::convertible_to<Loc>;
        { loc - std::declval<typename Loc::difference_type>() } -> std::convertible_to<Loc>;
        { *loc } -> std::convertible_to<typename Loc::reference>;
        { loc[std::declval<typename Loc::difference_type>()] } -> std::convertible_to<typename Loc::reference>;
        {
            loc.cacheLocation(std::declval<typename Loc::difference_type>())
        } -> std::convertible_to<typename Loc::cached_location_type>;
        { loc[std::declval<typename Loc::cached_location_type>()] } -> std::convertible_to<typename Loc::reference>;

        { loc.x() } -> std::convertible_to<typename Loc::x_iterator const &>;
        { loc.y() } -> std::convertible_to<typename Loc::y_iterator const &>;
        { loc.xAt(std::declval<typename Loc::difference_type>()) } -> std::convertible_to<typename Loc::x_iterator>;
        { loc.yAt(std::declval<typename Loc::difference_type>()) } -> std::convertible_to<typename Loc::y_iterator>;
        { loc.xyAt(std::declval<typename Loc::difference_type>()) } -> std::convertible_to<Loc>;
        {
            loc.xAt(std::declval<typename Loc::x_coordinate_type>(), std::declval<typename Loc::y_coordinate_type>())
        } -> std::convertible_to<typename Loc::x_iterator>;
        {
            loc.yAt(std::declval<typename Loc::x_coordinate_type>(), std::declval<typename Loc::y_coordinate_type>())
        } -> std::convertible_to<typename Loc::y_iterator>;
        {
            loc.xyAt(std::declval<typename Loc::x_coordinate_type>(), std::declval<typename Loc::y_coordinate_type>())
        } -> std::convertible_to<Loc>;
        {
            loc(std::declval<typename Loc::x_coordinate_type>(), std::declval<typename Loc::y_coordinate_type>())
        } -> std::convertible_to<typename Loc::reference>;
        {
            loc.cacheLocation(std::declval<typename Loc::x_coordinate_type>(),
                              std::declval<typename Loc::y_coordinate_type>())
        } -> std::convertible_to<typename Loc::cached_location_type>;

        { loc.is1DTraversable(std::declval<typename Loc::x_coordinate_type>()) } -> std::convertible_to<bool>;
        {
            loc.yDistanceTo(loc2, std::declval<typename Loc::x_coordinate_type>())
        } -> std::convertible_to<typename Loc::y_coordinate_type>;

        { loc.width() } -> std::convertible_to<typename Loc::x_coordinate_type &>;
        { loc.height() } -> std::convertible_to<typename Loc::y_coordinate_type &>;
        { cloc.width() } -> std::convertible_to<typename Loc::x_coordinate_type const &>;
        { cloc.height() } -> std::convertible_to<typename Loc::y_coordinate_type const &>;
    };

    template<typename Loc>
    concept IsPixelLocator = Is2DLocator<Loc> and requires {
        requires IsPixel<typename Loc::value_type>;
        requires IsPixel<typename Loc::reference>;
        requires IsPoint<typename Loc::difference_type>;
        requires IsPoint<typename Loc::point_type>;

        typename Loc::coordinate_type;
        requires std::same_as<typename Loc::x_coordinate_type, typename Loc::y_coordinate_type>;

        typename Loc::transposed_type;
    };

    template<typename Loc>
    concept IsPixelLocatorHasTransposedType = IsPixelLocator<Loc> and requires { typename Loc::transposed_type; };

    template<typename View>
    concept IsImageView = requires(View v, View const cv) {
        typename View::value_type;
        typename View::reference;
        typename View::difference_type;
        typename View::const_view;
        typename View::point_type;
        typename View::locator;
        typename View::iterator;
        typename View::reverse_iterator;
        typename View::size_type;
        typename View::x_iterator;
        typename View::y_iterator;
        typename View::x_coordinate_type;
        typename View::y_coordinate_type;
        typename View::xy_locator;

        requires IsPixel<typename View::value_type>;
        requires IsPoint<typename View::point_type>;
        requires IsPixelLocator<typename View::locator>;
        requires IsPixelLocator<typename View::xy_locator>;
        requires IsPixelIterator<typename View::iterator>;
        // requires IsPixelIterator<typename View::reverse_iterator>;
        requires IsPixelIterator<typename View::x_iterator>;
        requires IsPixelIterator<typename View::y_iterator>;

        requires std::constructible_from<View, typename View::point_type, typename View::locator const &>;
        requires std::constructible_from<View, typename View::x_coordinate_type, typename View::y_coordinate_type,
                                         typename View::locator const &>;

        { cv.size() } -> std::convertible_to<typename View::size_type>;
        { cv.width() } -> std::convertible_to<typename View::x_coordinate_type const &>;
        { v.width() } -> std::convertible_to<typename View::x_coordinate_type &>;
        { cv.height() } -> std::convertible_to<typename View::y_coordinate_type const &>;
        { v.height() } -> std::convertible_to<typename View::y_coordinate_type &>;

        { cv[std::declval<typename View::difference_type>()] } -> std::convertible_to<typename View::reference>;

        { cv.begin() } -> std::convertible_to<typename View::iterator>;
        { cv.end() } -> std::convertible_to<typename View::iterator>;
        { cv.rbegin() } -> std::convertible_to<typename View::reverse_iterator>;
        { cv.rend() } -> std::convertible_to<typename View::reverse_iterator>;
        { v.at(std::declval<typename View::point_type>()) } -> std::convertible_to<typename View::iterator>;
        {
            v.at(std::declval<typename View::x_coordinate_type>(), std::declval<typename View::y_coordinate_type>())
        } -> std::convertible_to<typename View::iterator>;

        { cv.is1DTraversable() } -> std::convertible_to<bool>;
        { cv(std::declval<typename View::point_type>()) } -> std::convertible_to<typename View::reference>;
        {
            cv(std::declval<typename View::x_coordinate_type>(), std::declval<typename View::y_coordinate_type>())
        } -> std::convertible_to<typename View::reference>;

        { cv.xAt(std::declval<typename View::point_type>()) } -> std::convertible_to<typename View::x_iterator>;
        {
            cv.xAt(std::declval<typename View::x_coordinate_type>(), std::declval<typename View::y_coordinate_type>())
        } -> std::convertible_to<typename View::x_iterator>;
        {
            cv.rowBegin(std::declval<typename View::y_coordinate_type>())
        } -> std::convertible_to<typename View::x_iterator>;
        {
            cv.rowEnd(std::declval<typename View::y_coordinate_type>())
        } -> std::convertible_to<typename View::x_iterator>;

        { cv.yAt(std::declval<typename View::point_type>()) } -> std::convertible_to<typename View::y_iterator>;
        {
            cv.yAt(std::declval<typename View::x_coordinate_type>(), std::declval<typename View::y_coordinate_type>())
        } -> std::convertible_to<typename View::y_iterator>;
        {
            cv.colBegin(std::declval<typename View::x_coordinate_type>())
        } -> std::convertible_to<typename View::y_iterator>;
        {
            cv.colEnd(std::declval<typename View::x_coordinate_type>())
        } -> std::convertible_to<typename View::y_iterator>;

        { cv.xyAt(std::declval<typename View::point_type>()) } -> std::convertible_to<typename View::xy_locator>;
        {
            cv.xyAt(std::declval<typename View::x_coordinate_type>(), std::declval<typename View::y_coordinate_type>())
        } -> std::convertible_to<typename View::xy_locator>;
    };

    template<typename V1, typename V2>
    concept IsImageViewsCompatible = IsImageView<V1> and IsImageView<V2> and
                                     IsPixelsCompatible<typename V1::value_type, typename V2::value_type>;

    template<typename T>
    concept IsImageContainer =
            std::default_initializable<T> and std::copyable<T> and std::movable<T> and std::equality_comparable<T> and
            requires(T t, T const ct) {
                typename T::allocator_type;
                typename T::view_type;
                typename T::const_view_type;
                typename T::point_type;
                typename T::value_type;
                typename T::x_coordinate_type;
                typename T::y_coordinate_type;

                requires IsImageView<typename T::view_type>;
                requires IsImageView<typename T::const_view_type>;
                requires IsPoint<typename T::point_type>;
                requires IsPixel<typename T::value_type>;

                {
                    t[std::declval<typename T::x_coordinate_type>(), std::declval<typename T::y_coordinate_type>()]
                } -> std::convertible_to<typename T::value_type &>;
                {
                    ct[std::declval<typename T::x_coordinate_type>(), std::declval<typename T::y_coordinate_type>()]
                } -> std::convertible_to<typename T::value_type const &>;
                { t[std::declval<typename T::point_type>()] } -> std::convertible_to<typename T::value_type &>;
                { ct[std::declval<typename T::point_type>()] } -> std::convertible_to<typename T::value_type const &>;
                { ct.toView() } -> std::convertible_to<typename T::view_type>;
            };
} // namespace mgil::inline concepts

namespace mgil::details {
    template<typename Range>
    struct infer_std_range_over_pixel;

    template<typename Range>
        requires std::ranges::range<std::ranges::range_value_t<Range>>
    struct infer_std_range_over_pixel<Range> {
        using type = std::ranges::range_value_t<std::ranges::range_value_t<Range>>;
    };

    template<typename Range>
        requires IsPixel<std::ranges::range_value_t<Range>>
    struct infer_std_range_over_pixel<Range> {
        using type = std::ranges::range_value_t<Range>;
    };

    template<typename Range>
    using infer_std_range_over_pixel_t = typename infer_std_range_over_pixel<Range>::type;
} // namespace mgil::details

// The main namespace
namespace mgil {
    // Color Spaces and Layouts
    struct red_color_t : color_space_component_tag {};
    struct green_color_t : color_space_component_tag {};
    struct blue_color_t : color_space_component_tag {};
    struct alpha_color_t : color_space_component_tag {};
    struct gray_color_t : color_space_component_tag {};
    struct cyan_color_t : color_space_component_tag {};
    struct magenta_color_t : color_space_component_tag {};
    struct yellow_color_t : color_space_component_tag {};
    struct black_color_t : color_space_component_tag {};

    // Color Space Models
    using gray_t = details::type_list<gray_color_t>;
    using rgb_t = details::type_list<red_color_t, green_color_t, blue_color_t>;
    using rgba_t = details::type_list<red_color_t, green_color_t, blue_color_t, alpha_color_t>;
    using cmyk_t = details::type_list<cyan_color_t, magenta_color_t, yellow_color_t, black_color_t>;

    // Color Space Layouts
    using gray_layout_t = ColorSpaceLayout<gray_t, details::index_list<0>>;
    using rgb_layout_t = ColorSpaceLayout<rgb_t, details::index_list<0, 1, 2>>;
    using bgr_layout_t = ColorSpaceLayout<rgb_t, details::index_list<2, 1, 0>>;
    using rgba_layout_t = ColorSpaceLayout<rgba_t, details::index_list<0, 1, 2, 3>>;
    using bgra_layout_t = ColorSpaceLayout<rgba_t, details::index_list<2, 1, 0, 3>>;
    using abgr_layout_t = ColorSpaceLayout<rgba_t, details::index_list<3, 2, 1, 0>>;
    using argb_layout_t = ColorSpaceLayout<rgba_t, details::index_list<3, 0, 1, 2>>;
    using cmyk_layout_t = ColorSpaceLayout<cmyk_t, details::index_list<0, 1, 2, 3>>;

#ifndef MGIL_NO_COMPILE_TIME_TESTING
    static_assert(IsColorSpace<gray_t> && IsColorSpace<rgb_t> and IsColorSpace<rgba_t> and IsColorSpace<cmyk_t>);
    static_assert(std::same_as<details::type_list<gray_color_t>, gray_layout_t::mapped_color_space>);
    static_assert(std::same_as<details::type_list<red_color_t, green_color_t, blue_color_t>,
                               rgb_layout_t::mapped_color_space>);
    static_assert(std::same_as<details::type_list<blue_color_t, green_color_t, red_color_t>,
                               bgr_layout_t::mapped_color_space>);
    static_assert(std::same_as<details::type_list<blue_color_t, green_color_t, red_color_t, alpha_color_t>,
                               bgra_layout_t::mapped_color_space>);
    static_assert(std::same_as<details::type_list<alpha_color_t, blue_color_t, green_color_t, red_color_t>,
                               abgr_layout_t::mapped_color_space>);
    static_assert(std::same_as<details::type_list<alpha_color_t, red_color_t, green_color_t, blue_color_t>,
                               argb_layout_t::mapped_color_space>);
    static_assert(std::same_as<details::type_list<cyan_color_t, magenta_color_t, yellow_color_t, black_color_t>,
                               cmyk_layout_t::mapped_color_space>);
#endif
} // namespace mgil

namespace mgil {
    template<IsChannel T, typename ChannelTraits<T>::value_type Min, typename ChannelTraits<T>::value_type Max>
    struct RescopedChannel;

    template<IsChannel T, typename ChannelTraits<T>::value_type Min, typename ChannelTraits<T>::value_type Max>
        requires std::integral<T> or std::floating_point<T>
    struct RescopedChannel<T, Min, Max> : channel_tag {
        T value;

        using value_type = T;
        using reference = T &;
        using pointer = T *;
        using const_reference = T const &;
        using const_pointer = T const *;
        static constexpr bool is_mutable = true;
        constexpr RescopedChannel() : value(Min) {
        }
        constexpr RescopedChannel(RescopedChannel const &that) = default;
        constexpr RescopedChannel(RescopedChannel &&that) noexcept = default;
        constexpr auto operator=(RescopedChannel const &that) -> RescopedChannel & = default;
        constexpr auto operator=(RescopedChannel &&that) noexcept -> RescopedChannel & = default;
        explicit constexpr RescopedChannel(T value_) : value(value_) {
            if (value < Min) {
                value = Min;
            }
            if (value > Max) {
                value = Max;
            }
        }
        template<typename U>
            requires std::convertible_to<U, T>
        explicit constexpr RescopedChannel(U value_) : value(static_cast<T>(value_)) {
            if (value < Min) {
                value = Min;
            }
            if (value > Max) {
                value = Max;
            }
        }
        static constexpr auto minValue() -> value_type {
            return Min;
        }
        static constexpr auto maxValue() -> value_type {
            return Max;
        }
        [[nodiscard]] constexpr auto getValue() const -> value_type {
            return value;
        }
        constexpr auto setValue(value_type v) -> void {
            value = v;
        }

        template<typename U>
            requires std::convertible_to<U, value_type>
        constexpr auto operator=(U const &that) -> RescopedChannel & {
            value = static_cast<value_type>(that);
            if (value > Max) {
                value = Max;
            }
            if (value < Min) {
                value = Min;
            }
            return *this;
        }

        constexpr operator value_type() const {
            return value;
        }

        friend constexpr auto operator+(RescopedChannel const &lhs, RescopedChannel const &rhs) {
            RescopedChannel result = lhs.value + rhs.value;
            if (result.value < Min) {
                result.value = Min;
            }
            if (result.value > Max) {
                result.value = Max;
            }
            return result;
        }
        friend constexpr auto operator-(RescopedChannel const &lhs, RescopedChannel const &rhs) {
            RescopedChannel result = lhs.value - rhs.value;
            if (result.value < Min) {
                result.value = Min;
            }
            if (result.value > Max) {
                result.value = Max;
            }
            return result;
        }
        friend constexpr auto operator*(RescopedChannel const &lhs, RescopedChannel const &rhs) {
            RescopedChannel result = lhs.value * rhs.value;
            if (result.value < Min) {
                result.value = Min;
            }
            if (result.value > Max) {
                result.value = Max;
            }
            return result;
        }
        friend constexpr auto operator/(RescopedChannel const &lhs, RescopedChannel const &rhs) {
            RescopedChannel result = lhs.value / rhs.value;
            if (result.value < Min) {
                result.value = Min;
            }
            if (result.value > Max) {
                result.value = Max;
            }
            return result;
        }
        friend constexpr auto operator+(RescopedChannel const &lhs, value_type const &rhs) {
            RescopedChannel result = lhs.value + rhs;
            if (result.value < Min) {
                result.value = Min;
            }
            if (result.value > Max) {
                result.value = Max;
            }
            return result;
        }
        friend constexpr auto operator-(RescopedChannel const &lhs, value_type const &rhs) {
            RescopedChannel result = lhs.value - rhs;
            if (result.value < Min) {
                result.value = Min;
            }
            if (result.value > Max) {
                result.value = Max;
            }
            return result;
        }
        friend constexpr auto operator*(RescopedChannel const &lhs, value_type const &rhs) {
            RescopedChannel result = lhs.value * rhs;
            if (result.value < Min) {
                result.value = Min;
            }
            if (result.value > Max) {
                result.value = Max;
            }
            return result;
        }
        friend constexpr auto operator/(RescopedChannel const &lhs, value_type const &rhs) {
            RescopedChannel result = lhs.value / rhs;
            if (result.value < Min) {
                result.value = Min;
            }
            if (result.value > Max) {
                result.value = Max;
            }
            return result;
        }
        friend constexpr auto operator+(value_type const &lhs, RescopedChannel const &rhs) {
            RescopedChannel result = lhs + rhs.value;
            if (result.value < Min) {
                result.value = Min;
            }
            if (result.value > Max) {
                result.value = Max;
            }
            return result;
        }
        friend constexpr auto operator-(value_type const &lhs, RescopedChannel const &rhs) {
            RescopedChannel result = lhs - rhs.value;
            if (result.value < Min) {
                result.value = Min;
            }
            if (result.value > Max) {
                result.value = Max;
            }
            return result;
        }
        friend constexpr auto operator*(value_type const &lhs, RescopedChannel const &rhs) {
            RescopedChannel result = lhs * rhs.value;
            if (result.value < Min) {
                result.value = Min;
            }
            if (result.value > Max) {
                result.value = Max;
            }
            return result;
        }
        friend constexpr auto operator/(value_type const &lhs, RescopedChannel const &rhs) {
            RescopedChannel result = lhs / rhs.value;
            if (result.value < Min) {
                result.value = Min;
            }
            if (result.value > Max) {
                result.value = Max;
            }
            return result;
        }

        constexpr auto operator+=(auto const &rhs) -> RescopedChannel & {
            return *this = *this + rhs;
        }
        constexpr auto operator-=(auto const &rhs) -> RescopedChannel & {
            return *this = *this - rhs;
        }
        constexpr auto operator*=(auto const &rhs) -> RescopedChannel & {
            return *this = *this * rhs;
        }
        constexpr auto operator/=(auto const &rhs) -> RescopedChannel & {
            return *this = *this / rhs;
        }

        friend constexpr auto operator<=>(RescopedChannel const &lhs, RescopedChannel const &rhs)
                -> std::strong_ordering {
            return lhs.value <=> rhs.value;
        }
        friend constexpr auto operator==(RescopedChannel const &lhs, RescopedChannel const &rhs) -> bool {
            return lhs.value == rhs.value;
        }
        friend constexpr auto operator!=(RescopedChannel const &lhs, RescopedChannel const &rhs) -> bool {
            return lhs.value != rhs.value;
        }

        friend constexpr auto operator<<(std::ostream &os, RescopedChannel const &rhs) -> std::ostream & {
            os << rhs.value;
            return os;
        }
    };

    template<IsChannel T, typename ChannelTraits<T>::value_type Min, typename ChannelTraits<T>::value_type Max>
        requires std::is_class_v<T>
    struct RescopedChannel<T, Min, Max> : T {
        static constexpr auto minValue() -> typename ChannelTraits<T>::value_type {
            return ChannelTraits<T>::minValue();
        }
        static constexpr auto maxValue() -> typename ChannelTraits<T>::value_type {
            return ChannelTraits<T>::maxValue();
        }
        [[nodiscard]] constexpr auto getValue() const -> typename ChannelTraits<T>::value_type {
            return getValue();
        }
        constexpr auto setValue(typename ChannelTraits<T>::value_type v) -> void {
            return setValue(v);
        }
    };

    using Float_01 = RescopedChannel<float, 0.0f, 1.0f>;
    using Double_01 = RescopedChannel<double, 0.0, 1.0>;
    using UInt8_0255 = RescopedChannel<std::uint8_t, 0, 255>;
    using UInt16_0255 = RescopedChannel<std::uint16_t, 0, 255>;
    using UInt32_0255 = RescopedChannel<std::uint32_t, 0, 255>;
    using UInt64_0255 = RescopedChannel<std::uint64_t, 0, 255>;

#ifndef MGIL_NO_COMPILE_TIME_TESTING
    static_assert(IsChannel<Float_01>);
    static_assert(IsChannel<Double_01>);
    static_assert(IsChannel<UInt8_0255>);
    static_assert(IsChannel<UInt16_0255>);
    static_assert(IsChannel<UInt32_0255>);
    static_assert(IsChannel<UInt64_0255>);
#endif
} // namespace mgil

namespace mgil {
    // Pixel class
    template<typename ChannelType, typename Layout>
        requires IsChannel<ChannelType> and IsLayout<Layout>
    class Pixel {
    public:
        using layout_type = Layout;
        using channel_type = ChannelType;
        using value_type = Pixel;
        using reference = value_type &;
        using const_reference = value_type const &;

    private:
        static constexpr std::size_t size = Layout::color_space::size;
        std::array<ChannelType, size> _components = {};

        template<std::size_t Index>
        static constexpr auto indexOfNthSemanticComponent() -> std::size_t {
            return details::find_type_v<typename layout_type::color_space::template NthType<Index>,
                                        typename layout_type::mapped_color_space>;
        }

        static constexpr auto add_sat_if_integral(auto lhs, auto rhs) {
            if constexpr (std::integral<decltype(lhs)> and std::integral<decltype(rhs)>) {
                return std::add_sat(lhs, rhs);
            }
            return lhs + rhs;
        }
        static constexpr auto sub_sat_if_integral(auto lhs, auto rhs) {
            if constexpr (std::integral<decltype(lhs)> and std::integral<decltype(rhs)>) {
                return std::sub_sat(lhs, rhs);
            }
            return lhs - rhs;
        }
        static constexpr auto mul_sat_if_integral(auto lhs, auto rhs) {
            if constexpr (std::integral<decltype(lhs)> and std::integral<decltype(rhs)>) {
                return std::mul_sat(lhs, rhs);
            }
            return lhs * rhs;
        }
        static constexpr auto div_sat_if_integral(auto lhs, auto rhs) {
            if constexpr (std::integral<decltype(lhs)> and std::integral<decltype(rhs)>) {
                return std::div_sat(lhs, rhs);
            }
            return lhs / rhs;
        }

    public:
        // Default constructor
        Pixel() = default;
        // Copy constructor
        Pixel(Pixel const &) = default;
        // Move constructor
        Pixel(Pixel &&) noexcept(std::is_nothrow_move_constructible_v<ChannelType>) = default;
        // Copy assignment
        auto operator=(Pixel const &) -> Pixel & = default;
        // Move assignment
        auto operator=(Pixel &&) noexcept(std::is_nothrow_move_assignable_v<ChannelType>) -> Pixel & = default;

        // Initialize the underlying array by setting all the elements v
        explicit constexpr Pixel(ChannelType v) : _components() {
            _components.fill(v);
        }

        // Initializing the underlying array by a pack
        template<typename... Ts>
            requires(std::convertible_to<Ts, ChannelType> && ...) and (sizeof...(Ts) == size)
        explicit constexpr Pixel(Ts... components) : _components{static_cast<ChannelType>(components)...} {
        }

        explicit constexpr Pixel(std::array<ChannelType, size> const &arr) : _components(arr) {
        }

        template<typename Range>
            requires std::ranges::range<Range> and std::convertible_to<std::ranges::range_value_t<Range>, ChannelType>
        explicit constexpr Pixel(Range &&range) {
            std::size_t i = 0;
            for (auto const elem: std::forward<Range>(range)) {
                _components[i] = elem;
                ++i;
            }
        }

        // Initializing from a compatible pixel
        template<typename SrcPixel>
            requires IsPixelsConvertible<SrcPixel, Pixel>
        explicit constexpr Pixel(SrcPixel const &src) : Pixel(convertPixelTo<SrcPixel, Pixel>(src)) {
        }

        template<typename SrcPixel, typename DstPixel>
            requires IsPixelsConvertible<SrcPixel, DstPixel>
        static constexpr auto convertPixelTo(SrcPixel const &src) -> DstPixel {
            using SrcChannelType = typename SrcPixel::channel_type;
            using DstChannelType = typename DstPixel::channel_type;
            using SrcLayoutType = typename SrcPixel::layout_type;
            using DstLayoutType = typename DstPixel::layout_type;
            DstPixel dst;
            if constexpr (std::same_as<typename SrcPixel::layout_type, DstLayoutType>) {
                for (std::size_t i = 0; i < size; i++) {
                    if constexpr (std::same_as<SrcChannelType, DstChannelType>) {
                        dst.set(i, src.get(i));
                    } else {
                        dst.set(i, ChannelTraits<SrcChannelType>::template convertTo<DstChannelType>(src.get(i)));
                    }
                }
            } else {
                details::mp::for_each<DstLayoutType::mapped_color_space::getMPVector()>([&]<details::mp::info meta> {
                    constexpr std::size_t src_index = details::find_type_v<details::mp::type_of<meta>,
                                                                           typename SrcLayoutType::mapped_color_space>;
                    constexpr std::size_t dst_index = details::find_type_v<details::mp::type_of<meta>,
                                                                           typename DstLayoutType::mapped_color_space>;
                    if constexpr (std::same_as<SrcChannelType, DstChannelType>) {
                        dst.set(dst_index, src.get(src_index));
                    } else {
                        dst.set(dst_index,
                                ChannelTraits<SrcChannelType>::template convertTo<DstChannelType>(src.get(src_index)));
                    }
                });
            }
            return dst;
        }

        template<typename Channel>
            requires IsChannel<Channel>
        constexpr auto castTo() const -> Pixel<Channel, layout_type> {
            Pixel<Channel, layout_type> result;
            for (std::size_t i = 0; i < size; i++) {
                result.get(i) = static_cast<Channel const>(_components[i]);
            }
            return result;
        }

        friend constexpr auto operator<=>(Pixel const &lhs, Pixel const &rhs) -> std::strong_ordering {
            return lhs._components <=> rhs._components;
        }
        friend constexpr auto operator==(Pixel const &lhs, Pixel const &rhs) -> bool {
            return lhs._components == rhs._components;
        }
        friend constexpr auto operator!=(Pixel const &lhs, Pixel const &rhs) -> bool {
            return lhs._components != rhs._components;
        }

        friend constexpr auto operator+(Pixel const &lhs, Pixel const &rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = add_sat_if_integral(lhs._components[i], rhs._components[i]);
            }
            return result;
        }
        friend constexpr auto operator-(Pixel const &lhs, Pixel const &rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = sub_sat_if_integral(lhs._components[i], rhs._components[i]);
            }
            return result;
        }
        friend constexpr auto operator*(Pixel const &lhs, Pixel const &rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = mul_sat_if_integral(lhs._components[i], rhs._components[i]);
            }
            return result;
        }
        friend constexpr auto operator/(Pixel const &lhs, Pixel const &rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = div_sat_if_integral(lhs._components[i], rhs._components[i]);
            }
            return result;
        }

        friend constexpr auto operator+(Pixel const &lhs, std::convertible_to<channel_type> auto rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = add_sat_if_integral(lhs._components[i], static_cast<channel_type>(rhs));
            }
            return result;
        }
        friend constexpr auto operator-(Pixel const &lhs, std::convertible_to<channel_type> auto rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = sub_sat_if_integral(lhs._components[i], static_cast<channel_type>(rhs));
            }
            return result;
        }
        friend constexpr auto operator*(Pixel const &lhs, std::convertible_to<channel_type> auto rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = mul_sat_if_integral(lhs._components[i], static_cast<channel_type>(rhs));
            }
            return result;
        }
        friend constexpr auto operator/(Pixel const &lhs, std::convertible_to<channel_type> auto rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = div_sat_if_integral(lhs._components[i], static_cast<channel_type>(rhs));
            }
            return result;
        }

        friend constexpr auto operator+(std::convertible_to<channel_type> auto lhs, Pixel const &rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = add_sat_if_integral(static_cast<channel_type>(lhs), rhs._components[i]);
            }
            return result;
        }
        friend constexpr auto operator-(std::convertible_to<channel_type> auto rhs, Pixel const &lhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = sub_sat_if_integral(static_cast<channel_type>(lhs), rhs._components[i]);
            }
            return result;
        }
        friend constexpr auto operator*(std::convertible_to<channel_type> auto lhs, Pixel const &rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = mul_sat_if_integral(static_cast<channel_type>(lhs), rhs._components[i]);
            }
            return result;
        }
        friend constexpr auto operator/(std::convertible_to<channel_type> auto lhs, Pixel const &rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = div_sat_if_integral(static_cast<channel_type>(lhs), rhs._components[i]);
            }
            return result;
        }

        constexpr auto operator+=(auto rhs) -> Pixel & {
            return *this = *this + rhs;
        }
        constexpr auto operator-=(auto rhs) -> Pixel & {
            return *this = *this - rhs;
        }
        constexpr auto operator*=(auto rhs) -> Pixel & {
            return *this = *this * rhs;
        }
        constexpr auto operator/=(auto rhs) -> Pixel & {
            return *this = *this / rhs;
        }

        [[nodiscard]] constexpr auto get(std::size_t index) noexcept -> ChannelType & {
            return _components[index];
        }
        [[nodiscard]] constexpr auto get(std::size_t index) const noexcept -> ChannelType const & {
            return _components[index];
        }
        template<std::size_t Index>
            requires(Index < size)
        [[nodiscard]] constexpr auto get() noexcept -> ChannelType & {
            return _components[Index];
        }
        template<std::size_t Index>
            requires(Index < size)
        [[nodiscard]] constexpr auto get() const noexcept -> ChannelType const & {
            return _components[Index];
        }
        template<std::size_t Index>
        [[nodiscard]] constexpr auto getSemantic() noexcept -> ChannelType & {
            return _components[indexOfNthSemanticComponent<Index>()];
        }
        template<std::size_t Index>
        [[nodiscard]] constexpr auto getSemantic() const noexcept -> ChannelType const & {
            return _components[indexOfNthSemanticComponent<Index>()];
        }
        template<typename Component>
            requires IsColorSpaceComponent<Component> and
                     details::contain_type_v<Component, typename layout_type::color_space>
        [[nodiscard]] constexpr auto get() noexcept -> ChannelType & {
            return _components[details::find_type_v<Component, typename layout_type::mapped_color_space>];
        }
        template<typename Component>
            requires IsColorSpaceComponent<Component> and
                     details::contain_type_v<Component, typename layout_type::color_space>
        [[nodiscard]] constexpr auto get() const noexcept -> ChannelType const & {
            return _components[details::find_type_v<Component, typename layout_type::mapped_color_space>];
        }
        constexpr auto set(std::size_t index, ChannelType v) -> void {
            _components[index] = v;
        }
        constexpr auto operator[](std::size_t const index) const noexcept -> ChannelType const & {
            return _components[index];
        }
        constexpr auto operator[](std::size_t const index) -> ChannelType & {
            return _components[index];
        }
        static constexpr auto getSize() noexcept -> std::size_t {
            return size;
        }

        friend auto operator<<(std::ostream &out, Pixel const &pixel) -> std::ostream & {
            out << std::string{"{"};
            if (size > 1) {
                for (std::size_t i = 0; i < getSize() - 1; i++) {
                    out << pixel.get(i) << ",";
                }
                out << pixel.get(getSize() - 1) << "}";
            } else {
                out << pixel.get(0) << "}";
            }
            return out;
        }
    };

    template<std::size_t Index, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> const &pixel) noexcept -> Channel const & {
        return pixel.get(Index);
    }
    template<std::size_t Index, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> &pixel) noexcept -> Channel & {
        return pixel.get(Index);
    }
    template<std::size_t Index, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> const &&pixel) noexcept -> Channel const && {
        return std::move(pixel.get(Index));
    }
    template<std::size_t Index, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> &&pixel) noexcept -> Channel && {
        return std::move(pixel.get(Index));
    }
    template<typename Component, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> const &pixel) noexcept -> Channel const & {
        return pixel.template get<Component>();
    }
    template<typename Component, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> &pixel) noexcept -> Channel & {
        return pixel.template get<Component>();
    }
    template<typename Component, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> const &&pixel) noexcept -> Channel const && {
        return std::move(pixel.template get<Component>());
    }
    template<typename Component, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> &&pixel) noexcept -> Channel && {
        return std::move(pixel.template get<Component>());
    }

    template<typename T>
        requires IsPixel<T> and IsLayoutCompatible<typename T::layout_type, rgb_layout_t>
    struct traits::PixelTraits<T> {
        template<typename U>
            requires IsPixel<U> and IsLayoutCompatible<typename U::layout_type, rgba_layout_t>
        static constexpr auto convertTo(T src) -> U {
            U result;
            result.template get<red_color_t> = src.template get<red_color_t>();
            result.template get<green_color_t> = src.template get<green_color_t>();
            result.template get<blue_color_t> = src.template get<blue_color_t>();
            result.template get<alpha_color_t> = 100;
            return result;
        }
        template<typename U>
            requires IsPixel<U> and IsLayoutCompatible<typename U::layout_type, gray_layout_t>
        static constexpr auto convertTo(T src) -> U {
            auto red = src.template get<red_color_t>();
            auto green = src.template get<green_color_t>();
            auto blue = src.template get<blue_color_t>();
            return U((red + green + blue) / typename U::channel_type(3));
        }
        template<typename U>
            requires IsPixel<U> and IsLayoutCompatible<typename U::layout_type, cmyk_layout_t>
        static constexpr auto convertTo(T src) -> U {
            auto red = src.template get<red_color_t>() / 255.0;
            auto green = src.template get<green_color_t>() / 255.0;
            auto blue = src.template get<blue_color_t>() / 255.0;
            auto cyan = 1.0 - red;
            auto magenta = 1.0 - green;
            auto yellow = 1.0 - blue;
            auto black = std::min(cyan, std::min(magenta, yellow));
            cyan = (cyan - black) / (1.0 - black);
            magenta = (magenta - black) / (1.0 - black);
            yellow = (yellow - black) / (1.0 - black);
            if constexpr (std::is_integral_v<typename U::layout_type>) {
                U result;
                result.template get<cyan_color_t> = std::round(cyan * 100.0);
                result.template get<magenta_color_t> = std::round(magenta * 100.0);
                result.template get<yellow_color_t> = std::round(yellow * 100.0);
                result.template get<black_color_t> = std::round(black * 100.0);
                return result;
            } else {
                U result;
                result.template get<cyan_color_t> = cyan;
                result.template get<magenta_color_t> = magenta;
                result.template get<yellow_color_t> = yellow;
                result.template get<black_color_t> = black;
                return result;
            }
        }
    };

    template<typename T>
        requires IsPixel<T> and IsLayoutCompatible<typename T::layout_type, rgba_layout_t>
    struct traits::PixelTraits<T> {
        template<typename U>
            requires IsPixel<U> and IsLayoutCompatible<typename U::layout_type, rgb_layout_t>
        static constexpr auto convertTo(T src) -> U {
            U result;
            result.template get<red_color_t> = src.template get<red_color_t>();
            result.template get<green_color_t> = src.template get<green_color_t>();
            result.template get<blue_color_t> = src.template get<blue_color_t>();
            return result;
        }
        template<typename U>
            requires IsPixel<U> and IsLayoutCompatible<typename U::layout_type, gray_layout_t>
        static constexpr auto convertTo(T src) -> U {
            auto red = src.template get<red_color_t>();
            auto green = src.template get<green_color_t>();
            auto blue = src.template get<blue_color_t>();
            return U((red + green + blue) / typename U::channel_type(3));
        }
        template<typename U>
            requires IsPixel<U> and IsLayoutCompatible<typename U::layout_type, cmyk_layout_t>
        static constexpr auto convertTo(T src) -> U {
            auto red = src.template get<red_color_t>() / 255.0;
            auto green = src.template get<green_color_t>() / 255.0;
            auto blue = src.template get<blue_color_t>() / 255.0;
            auto cyan = 1.0 - red;
            auto magenta = 1.0 - green;
            auto yellow = 1.0 - blue;
            auto black = std::min(cyan, std::min(magenta, yellow));
            cyan = (cyan - black) / (1.0 - black);
            magenta = (magenta - black) / (1.0 - black);
            yellow = (yellow - black) / (1.0 - black);
            if constexpr (std::is_integral_v<typename U::layout_type>) {
                U result;
                result.template get<cyan_color_t> = std::round(cyan * 100.0);
                result.template get<magenta_color_t> = std::round(magenta * 100.0);
                result.template get<yellow_color_t> = std::round(yellow * 100.0);
                result.template get<black_color_t> = std::round(black * 100.0);
                return result;
            } else {
                U result;
                result.template get<cyan_color_t> = cyan;
                result.template get<magenta_color_t> = magenta;
                result.template get<yellow_color_t> = yellow;
                result.template get<black_color_t> = black;
                return result;
            }
        }
    };

    template<typename T>
        requires IsPixel<T> and IsLayoutCompatible<typename T::layout_type, gray_layout_t>
    struct traits::PixelTraits<T> {
        template<typename U>
            requires IsPixel<U> and IsLayoutCompatible<typename U::layout_type, rgb_layout_t>
        static constexpr auto convertTo(T src) -> U {
            U result;
            result.template get<red_color_t> = src.template get<gray_color_t>();
            result.template get<green_color_t> = src.template get<gray_color_t>();
            result.template get<blue_color_t> = src.template get<gray_color_t>();
            return result;
        }
    };

    template<typename T>
        requires IsPixel<T> and IsLayoutCompatible<typename T::layout_type, cmyk_layout_t>
    struct traits::PixelTraits<T> {
        template<typename U>
            requires IsPixel<U> and IsLayoutCompatible<typename U::layout_type, rgb_layout_t>
        static constexpr auto convertTo(T src) -> U {
            using dst_channel_type = typename U::channel_type;
            auto cyan = src.template get<cyan_color_t>();
            auto magenta = src.template get<magenta_color_t>();
            auto yellow = src.template get<yellow_color_t>();
            auto black = src.template get<black_color_t>();
            auto red = dst_channel_type(255) * (1.0 - cyan) * (1.0 - black);
            auto green = dst_channel_type(255) * (1.0 - magenta) * (1.0 - black);
            auto blue = dst_channel_type(255) * (1.0 - yellow) * (1.0 - black);
            U result;
            result.template get<red_color_t> = red;
            result.template get<green_color_t> = green;
            result.template get<blue_color_t> = blue;
            return result;
        }
        template<typename U>
            requires IsPixel<U> and IsLayoutCompatible<typename U::layout_type, rgba_layout_t>
        static constexpr auto convertTo(T src) -> U {
            using dst_channel_type = typename U::channel_type;
            auto cyan = src.template get<cyan_color_t>();
            auto magenta = src.template get<magenta_color_t>();
            auto yellow = src.template get<yellow_color_t>();
            auto black = src.template get<black_color_t>();
            auto red = dst_channel_type(255) * (1.0 - cyan) * (1.0 - black);
            auto green = dst_channel_type(255) * (1.0 - magenta) * (1.0 - black);
            auto blue = dst_channel_type(255) * (1.0 - yellow) * (1.0 - black);
            U result;
            result.template get<red_color_t> = red;
            result.template get<green_color_t> = green;
            result.template get<blue_color_t> = blue;
            result.template get<alpha_color_t> = 255;
            return result;
        }
    };
} // namespace mgil

template<typename Channel, typename Layout>
struct std::tuple_size<mgil::Pixel<Channel, Layout>>
    : std::integral_constant<std::size_t, mgil::Pixel<Channel, Layout>::getSize()> {};
template<std::size_t Index, typename Channel, typename Layout>
struct std::tuple_element<Index, mgil::Pixel<Channel, Layout>> {
    using type = Channel;
};
template<typename Channel, typename Layout>
    requires mgil::IsChannel<Channel> and mgil::IsLayout<Layout>
struct std::formatter<mgil::Pixel<Channel, Layout>> {
    constexpr auto parse(std::format_parse_context &ctx) {
        return ctx.begin();
    }
    auto format(mgil::Pixel<Channel, Layout> const &pixel, std::format_context &ctx) const {
        std::format_to(ctx.out(), "{{");
        if (mgil::Pixel<Channel, Layout>::getSize() > 1) {
            for (std::size_t i = 0; i < mgil::Pixel<Channel, Layout>::getSize() - 1; i++) {
                std::format_to(ctx.out(), "{},", pixel.get(i));
            }
            std::format_to(ctx.out(), "{}", pixel.get(mgil::Pixel<Channel, Layout>::getSize() - 1));
        } else {
            std::format_to(ctx.out(), "{}", pixel.get(0));
        }
        std::format_to(ctx.out(), "}}");
        return ctx.out();
    }
};
template<typename Channel>
    requires mgil::IsChannel<Channel> and std::is_class_v<Channel>
struct std::formatter<Channel> {
    constexpr auto parse(std::format_parse_context &ctx) {
        return ctx.begin();
    }
    auto format(Channel const &channel, std::format_context &ctx) const {
        return std::format_to(ctx.out(), "{}", mgil::ChannelTraits<Channel>::getValue(channel));
    }
};

namespace mgil {
#ifndef MGIL_NO_COMPILE_TIME_TESTING
    static_assert(IsPixel<Pixel<int, rgb_layout_t>>);
    static_assert(IsPixelsConvertible<Pixel<int, abgr_layout_t>, Pixel<int, bgra_layout_t>>);
    static_assert(Pixel<int, rgba_layout_t>(1, 2, 3, 4) ==
                  Pixel<int, rgba_layout_t>(Pixel<int, bgra_layout_t>(3, 2, 1, 4)));
    static_assert(Pixel<int, rgba_layout_t>(1, 2, 3, 4).get<blue_color_t>() == 3);
    static_assert(Pixel<int, rgba_layout_t>(1, 2, 3, 4).get(1) == 2);
    static_assert(Pixel<int, abgr_layout_t>(1, 2, 3, 4).getSemantic<1>() == 3);
    static_assert([] {
        Pixel<int, rgba_layout_t> pixel(1, 2, 3, 4);
        auto [r, g, b, a] = pixel;
        return r == 1 and g == 2 and b == 3 and a == 4;
    }());
#endif
} // namespace mgil

// Point class
namespace mgil {
    template<typename ValueType>
        requires std::is_arithmetic_v<ValueType>
    class Point {
    private:
        ValueType x_ = {};
        ValueType y_ = {};

    public:
        using value_type = ValueType;

        constexpr Point() = default;
        constexpr Point(Point const &that) : x_(that.x_), y_(that.y_) {
        }
        constexpr Point(Point &&that) noexcept : x_(std::move(that.x_)), y_(std::move(that.y_)) {
        }
        constexpr Point(value_type x, value_type y) : x_(x), y_(y) {
        }
        constexpr Point(value_type v) : x_(v), y_(v) {
        }
        constexpr auto operator=(Point const &that) -> Point & = default;
        constexpr auto operator=(Point &&that) noexcept -> Point & {
            this->x_ = std::move(that.x_);
            this->y_ = std::move(that.y_);
            return *this;
        }

        constexpr auto x() noexcept -> value_type & {
            return x_;
        }
        constexpr auto y() noexcept -> value_type & {
            return y_;
        }
        [[nodiscard]] constexpr auto x() const noexcept -> value_type const & {
            return x_;
        }
        [[nodiscard]] constexpr auto y() const noexcept -> value_type const & {
            return y_;
        }

        friend constexpr auto operator+(Point const &lhs, Point const &rhs) -> Point {
            return Point(lhs.x_ + rhs.x_, lhs.y_ + rhs.y_);
        }
        friend constexpr auto operator-(Point const &lhs, Point const &rhs) -> Point {
            return Point(lhs.x_ - rhs.x_, lhs.y_ - rhs.y_);
        }
        friend constexpr auto operator*(Point const &lhs, Point const &rhs) -> Point {
            return Point(lhs.x_ * rhs.x_, lhs.y_ * rhs.y_);
        }
        friend constexpr auto operator/(Point const &lhs, Point const &rhs) -> Point {
            return Point(lhs.x_ / rhs.x_, lhs.y_ / rhs.y_);
        }
        friend constexpr auto operator+(Point const &lhs, details::IsArithmetic auto rhs) -> Point {
            return Point(lhs.x_ + rhs, lhs.y_ + rhs);
        }
        friend constexpr auto operator-(Point const &lhs, details::IsArithmetic auto rhs) -> Point {
            return Point(lhs.x_ - rhs, lhs.y_ - rhs);
        }
        friend constexpr auto operator*(Point const &lhs, details::IsArithmetic auto rhs) -> Point {
            return Point(lhs.x_ * rhs, lhs.y_ * rhs);
        }
        friend constexpr auto operator/(Point const &lhs, details::IsArithmetic auto rhs) -> Point {
            return Point(lhs.x_ / rhs, lhs.y_ / rhs);
        }
        friend constexpr auto operator+(details::IsArithmetic auto rhs, Point const &lhs) -> Point {
            return Point(lhs.x_ + rhs, lhs.y_ + rhs);
        }
        friend constexpr auto operator-(details::IsArithmetic auto rhs, Point const &lhs) -> Point {
            return Point(lhs.x_ - rhs, lhs.y_ - rhs);
        }
        friend constexpr auto operator*(details::IsArithmetic auto rhs, Point const &lhs) -> Point {
            return Point(lhs.x_ * rhs, lhs.y_ * rhs);
        }
        friend constexpr auto operator/(details::IsArithmetic auto rhs, Point const &lhs) -> Point {
            return Point(lhs.x_ / rhs, lhs.y_ / rhs);
        }

        friend constexpr auto operator+=(Point &lhs, Point const &rhs) -> Point & {
            return lhs = Point(lhs.x_ + rhs.x_, lhs.y_ + rhs.y_);
        }
        friend constexpr auto operator-=(Point &lhs, Point const &rhs) -> Point & {
            return lhs = Point(lhs.x_ - rhs.x_, lhs.y_ - rhs.y_);
        }
        friend constexpr auto operator*=(Point &lhs, Point const &rhs) -> Point & {
            return lhs = Point(lhs.x_ * rhs.x_, lhs.y_ * rhs.y_);
        }
        friend constexpr auto operator/=(Point &lhs, Point const &rhs) -> Point & {
            return lhs = Point(lhs.x_ / rhs.x_, lhs.y_ / rhs.y_);
        }
        friend constexpr auto operator+=(Point &lhs, details::IsArithmetic auto rhs) -> Point & {
            return lhs = Point(lhs.x_ + rhs, lhs.y_ + rhs);
        }
        friend constexpr auto operator-=(Point &lhs, details::IsArithmetic auto rhs) -> Point & {
            return lhs = Point(lhs.x_ - rhs, lhs.y_ - rhs);
        }
        friend constexpr auto operator*=(Point &lhs, details::IsArithmetic auto rhs) -> Point & {
            return lhs = Point(lhs.x_ * rhs, lhs.y_ * rhs);
        }
        friend constexpr auto operator/=(Point &lhs, details::IsArithmetic auto rhs) -> Point & {
            return lhs = Point(lhs.x_ / rhs, lhs.y_ / rhs);
        }

        friend constexpr auto operator==(Point const &lhs, Point const &rhs) -> bool {
            return lhs.x_ == rhs.x_ and lhs.y_ == rhs.y_;
        }
        friend constexpr auto operator<=>(Point const &lhs, Point const &rhs) -> std::strong_ordering
            requires std::same_as<std::strong_ordering,
                                  decltype(std::declval<value_type>() <=> std::declval<value_type>())>
        {
            if (lhs.x_ < rhs.x_ or (lhs.x_ == rhs.x_ and lhs.y_ < rhs.y_)) {
                return std::strong_ordering::less;
            }
            if (lhs.x_ > rhs.x_ and (lhs.x_ == rhs.x_ and lhs.y_ > rhs.y_)) {
                return std::strong_ordering::greater;
            }
            return std::strong_ordering::equal;
        }
        friend constexpr auto operator<=>(Point const &lhs, Point const &rhs) -> std::weak_ordering
            requires std::same_as<std::weak_ordering,
                                  decltype(std::declval<value_type>() <=> std::declval<value_type>())>
        {
            if (lhs.x_ < rhs.x_ or (lhs.x_ == rhs.x_ and lhs.y_ < rhs.y_)) {
                return std::weak_ordering::less;
            }
            if (lhs.x_ > rhs.x_ and (lhs.x_ == rhs.x_ and lhs.y_ > rhs.y_)) {
                return std::weak_ordering::greater;
            }
            return std::weak_ordering::equivalent;
        }
        friend constexpr auto operator<=>(Point const &lhs, Point const &rhs) -> std::partial_ordering
            requires std::same_as<std::partial_ordering,
                                  decltype(std::declval<value_type>() <=> std::declval<value_type>())>
        {
            if (lhs.x_ == rhs.x_ and lhs.y_ == rhs.y_) {
                return std::partial_ordering::equivalent;
            }
            if (lhs.x_ < rhs.x_ or (lhs.x_ == rhs.x_ and lhs.y_ < rhs.y_)) {
                return std::partial_ordering::less;
            }
            if (lhs.x_ > rhs.x_ and (lhs.x_ == rhs.x_ and lhs.y_ > rhs.y_)) {
                return std::partial_ordering::greater;
            }
            return std::partial_ordering::unordered;
        }

        friend auto operator<<(std::ostream &os, Point const &rhs) -> std::ostream & {
            os << "Point(" << rhs.x_ << "," << rhs.y_ << ")";
            return os;
        }
    };
    template<std::size_t Index, typename ValueType>
        requires(Index == 0 or Index == 1)
    constexpr auto get(Point<ValueType> const &point) -> ValueType const & {
        if constexpr (Index == 0) {
            return point.x();
        }
        return point.y();
    }
    template<std::size_t Index, typename ValueType>
        requires(Index == 0 or Index == 1)
    constexpr auto get(Point<ValueType> &point) -> ValueType & {
        if constexpr (Index == 0) {
            return point.x();
        }
        return point.y();
    }
    template<std::size_t Index, typename ValueType>
        requires(Index == 0 or Index == 1)
    constexpr auto get(Point<ValueType> const &&point) -> ValueType const && {
        if constexpr (Index == 0) {
            return std::move(point.x());
        }
        return std::move(point.y());
    }
    template<std::size_t Index, typename ValueType>
        requires(Index == 0 or Index == 1)
    constexpr auto get(Point<ValueType> &&point) -> ValueType && {
        if constexpr (Index == 0) {
            return std::move(point.x());
        }
        return std::move(point.y());
    }
} // namespace mgil

template<typename ValueType>
struct std::tuple_size<mgil::Point<ValueType>> : std::integral_constant<std::size_t, 2> {};
template<std::size_t Index, typename ValueType>
    requires(Index == 0 or Index == 1)
struct std::tuple_element<Index, mgil::Point<ValueType>> {
    using type = ValueType;
};

template<typename ValueType>
struct std::formatter<mgil::Point<ValueType>> {
    constexpr auto parse(std::format_parse_context &ctx) {
        return ctx.begin();
    }
    auto format(mgil::Point<ValueType> const &point, std::format_context &ctx) const {
        return std::format_to(ctx.out(), "Point({},{})", point.x(), point.y());
    }
};

namespace mgil {
#ifndef MGIL_NO_COMPILE_TIME_TESTING
    static_assert(IsPoint<Point<int>>);
    static_assert([] {
        constexpr Point point{114, 514};
        auto [x, y] = point;
        return x == 114 and y == 514;
    }());
    static_assert([] {
        constexpr Point point1{2, 4};
        constexpr Point point2{3, 5};
        constexpr Point point3 = point1 * point2;
        auto [x, y] = point3;
        return x == 6 and y == 20;
    }());
#endif
} // namespace mgil

// Iterators
namespace mgil {
    // An iterator that remembers its current X and Y position and invokes a function object with it upon dereferencing.
    // Conforms to IsPixelIterator and IsStepIterator concepts
    template<typename Deref = identity_deref_adaptor<Pixel<int, gray_layout_t>>, bool IsTransposed = false,
             bool IsVirtual = true, typename PointType = Point<std::ptrdiff_t>>
        requires IsPoint<PointType> and IsPixelDereferenceAdaptor<Deref>
    struct position_iterator : pixel_iterator_tag,
                               details::iterator_facade<position_iterator<Deref, IsTransposed, IsVirtual, PointType>> {
    public:
        using const_iterator = position_iterator const;
        using value_type = typename Deref::value_type;
        using difference_type = std::ptrdiff_t;
        using point_type = PointType;
        using deref_type = Deref;
        using transposed_type = position_iterator<Deref, not IsTransposed, IsVirtual, PointType>;
        using size_type = std::size_t;
        using x_coordinate_type = std::ptrdiff_t;
        using y_coordinate_type = std::ptrdiff_t;

        static constexpr bool is_mutable = true;
        static constexpr bool transposed = IsTransposed;

    private:
        PointType pos_ = {};
        PointType step_ = {};
        Deref deref_ = {};
        x_coordinate_type width_ = 0;
        y_coordinate_type height_ = 0;
        value_type *underlying_ptr_ = nullptr;
        std::mdspan<value_type, std::dextents<std::size_t, 2>> underlying_;

    public:
        template<typename NewDeref>
            requires IsPixelDereferenceAdaptor<NewDeref>
        struct add_deref {
            using type = position_iterator<deref_compose<NewDeref, Deref>, IsTransposed, IsVirtual, PointType>;

            static constexpr auto make(position_iterator const &it, NewDeref const &new_deref) -> type {
                return type(it.pos(), it.step(), deref_compose<NewDeref, Deref>{});
            }
        };

        explicit constexpr position_iterator(PointType const &pos = {0, 0}, PointType const &step = {1, 1},
                                             Deref const &deref = {}, x_coordinate_type const width = 0,
                                             y_coordinate_type const height = 0, value_type *storage = nullptr) :
            pos_{pos}, step_{step}, deref_{deref}, width_{width}, height_{height}, underlying_ptr_{storage},
            underlying_{storage, width, height} {
        }
        constexpr position_iterator(position_iterator const &that) :
            pos_{that.pos_}, step_{that.step_}, deref_{that.deref_}, width_{that.width_}, height_{that.height_},
            underlying_ptr_{that.underlying_ptr_}, underlying_{that.underlying_} {
        }

        constexpr auto operator=(position_iterator const &that) -> position_iterator & {
            if (this == std::addressof(that)) {
                return *this;
            }
            pos_ = that.pos_;
            step_ = that.step_;
            deref_ = that.deref_;
            width_ = that.width_;
            height_ = that.height_;
            underlying_ptr_ = that.underlying_ptr_;
            underlying_ = that.underlying_;
            return *this;
        }

        constexpr auto pos() const -> PointType const & {
            return pos_;
        }
        constexpr auto pos() -> PointType & {
            return pos_;
        }
        constexpr auto step() const -> PointType const & {
            return step_;
        }
        constexpr auto derefFn() const {
            return deref_;
        }

        constexpr auto setStep(difference_type step) -> void {
            step_ = step;
        }

        constexpr auto dereference() const {
            if constexpr (details::is_specialization_of_v<deref_type, identity_deref_adaptor>) {
                return underlying_[pos_.x(), pos_.y()];
            } else {
                if constexpr (IsVirtual) {
                    return deref_(pos_);
                } else {
                    return deref_(pos_, underlying_[pos_.x(), pos_.y()]);
                }
            }
        }
        constexpr auto increment() -> void {
            pos_ += step_;
        }
        constexpr auto decrement() -> void {
            pos_ -= step_;
        }
        constexpr auto advance(difference_type d) -> void {
            pos_ += d * step_;
        }
        constexpr auto distance_to(position_iterator const &that) const -> difference_type {
            return ((that.pos_ - this->pos_) / step_).x();
        }
        constexpr auto equal_to(position_iterator const &that) const -> bool {
            if constexpr (requires(Deref const deref1, Deref const deref2) { deref1 == deref2; }) {
                if constexpr (IsVirtual) {
                    return pos_ == that.pos_ and step_ == that.step_ and deref_ == that.deref_ and
                           width_ == that.width_ and height_ == that.height_;
                } else {
                    return pos_ == that.pos_ and step_ == that.step_ and deref_ == that.deref_ and
                           width_ == that.width_ and height_ == that.height_ and
                           underlying_ptr_ == that.underlying_ptr_;
                }
            } else {
                if constexpr (IsVirtual) {
                    return pos_ == that.pos_ and step_ == that.step_ and width_ == that.width_ and
                           height_ == that.height_;
                } else {
                    return pos_ == that.pos_ and step_ == that.step_ and width_ == that.width_ and
                           height_ == that.height_ and underlying_ptr_ == that.underlying_ptr_;
                }
            }
        }

        constexpr auto width() -> x_coordinate_type & {
            return width_;
        }
        constexpr auto height() -> y_coordinate_type & {
            return height_;
        }
        [[nodiscard]] constexpr auto width() const -> x_coordinate_type const & {
            return width_;
        }
        [[nodiscard]] constexpr auto height() const -> y_coordinate_type const & {
            return height_;
        }
    };

#ifndef MGIL_NO_COMPILE_TIME_TESTING
    struct test_deref_adaptor
        : deref_base<test_deref_adaptor, Pixel<int, gray_layout_t>, Pixel<int, gray_layout_t> const,
                     Pixel<int, gray_layout_t> const &, Point<std::ptrdiff_t>, false> {
        constexpr auto operator()(Point<std::ptrdiff_t> point) const -> Pixel<int, gray_layout_t> {
            return Pixel<int, gray_layout_t>(point.x() + point.y());
        }
        constexpr auto operator=(test_deref_adaptor const &that) -> test_deref_adaptor & = default;
    };
    static_assert(IsPixelIterator<position_iterator<test_deref_adaptor>>);
    static_assert(IsStepIterator<position_iterator<test_deref_adaptor>>);
    static_assert(IsPixelIteratorHasTransposedType<position_iterator<test_deref_adaptor>>);
#endif

    // An iterator traverses a 2D image and provides random access to 1D rows.
    // Conforms only to IsPixelIterator concept since there are no steps.
    template<typename Loc>
        requires IsPixelLocator<Loc>
    struct two_dimensional_iterator : pixel_iterator_tag, details::iterator_facade<two_dimensional_iterator<Loc>> {
    public:
        using parent_type = details::iterator_facade<two_dimensional_iterator<Loc>>;
        using value_type = typename Loc::value_type;
        using difference_type = std::ptrdiff_t;
        using point_type = typename Loc::point_type;
        using size_type = std::size_t;
        using x_coordinate_type = std::ptrdiff_t;
        using y_coordinate_type = std::ptrdiff_t;
        using const_iterator = two_dimensional_iterator<typename Loc::const_locator>;

        static constexpr bool is_mutable = true;

    private:
        point_type pos_ = {};
        x_coordinate_type width_ = 0;
        y_coordinate_type height_ = 0;
        Loc loc_;

    public:
        constexpr two_dimensional_iterator() = default;
        explicit constexpr two_dimensional_iterator(Loc const &loc, x_coordinate_type const width,
                                                    y_coordinate_type const height, x_coordinate_type const x,
                                                    y_coordinate_type const y) :
            pos_{point_type{x, y}}, width_{width}, height_{height}, loc_{loc} {
            loc_.pos().x() = x;
            loc_.pos().y() = y;
        }
        constexpr two_dimensional_iterator(two_dimensional_iterator const &that) :
            pos_{that.pos_}, width_{that.width_}, height_{that.height_}, loc_{that.loc_} {
        }

        constexpr auto operator=(two_dimensional_iterator const &that) -> two_dimensional_iterator & {
            pos_ = that.pos_;
            width_ = that.width_;
            height_ = that.height_;
            loc_ = that.loc_;
            return *this;
        }

        constexpr auto pos() const -> point_type const & {
            return pos_;
        }

        constexpr auto dereference() const {
            return *loc_;
        }
        constexpr auto increment() -> void {
            ++pos_.x();
            ++loc_.pos().x();
            if (pos_.x() >= width_) {
                pos_.x() = 0;
                ++pos_.y();
                loc_.pos() += point_type{-width_, 1};
            }
        }
        constexpr auto decrement() -> void {
            --pos_.x();
            --loc_.pos().x();
            if (pos_.x() < 0) {
                pos_.x() = width_ - 1;
                --pos_.y();
                loc_.pos() += point_type{width_, -1};
            }
        }
        constexpr auto advance(difference_type d) -> void {
            if (width_ == 0) {
                return;
            }
            point_type delta = {};
            if (pos_.x() + d >= 0) {
                delta.x() = (pos_.x() + static_cast<std::ptrdiff_t>(d)) % width_ - pos_.x();
                delta.y() = (pos_.x() + static_cast<std::ptrdiff_t>(d)) / width_;
            } else {
                delta.x() = (pos_.x() + static_cast<std::ptrdiff_t>(d) * (1 - width_)) % width_ - pos_.x();
                delta.y() = -(width_ - pos_.x() - static_cast<std::ptrdiff_t>(d) - 1) / width_;
            }
            pos_ += delta;
            loc_.pos() += delta;
        }
        constexpr auto distance_to(two_dimensional_iterator const &that) const -> difference_type {
            if (width_ == 0) {
                return 0;
            }
            return (that.pos().y() - pos_.y()) * width_ + (that.pos().x() - pos_.x());
        }
        constexpr auto equal_to(two_dimensional_iterator const &that) const -> bool {
            assert(width_ == that.width_);
            return pos_ == that.pos_ and loc_ == that.loc_;
        }

        constexpr auto width() -> x_coordinate_type & {
            return width_;
        }
        constexpr auto height() -> y_coordinate_type & {
            return height_;
        }
        [[nodiscard]] constexpr auto width() const -> x_coordinate_type const & {
            return width_;
        }
        [[nodiscard]] constexpr auto height() const -> y_coordinate_type const & {
            return height_;
        }
    };
} // namespace mgil

// Locators
namespace mgil {
    // A 2D locator over a virtual image
    // Invokes a given function object passing its coordinates upon dereferencing
    // Conforms to IsPixelLocator, IsPixelLocatorHasTransposedType concepts
    template<typename Deref, bool IsTransposed = false, bool IsVirtual = true,
             typename PointType = Point<std::ptrdiff_t>>
        requires IsPixelDereferenceAdaptor<Deref> and IsPoint<PointType>
    class position_locator
        : public details::position_locator_base<position_locator<Deref, IsTransposed, IsVirtual, PointType>,
                                                position_iterator<Deref, IsTransposed, IsVirtual, PointType>,
                                                position_iterator<Deref, IsTransposed, IsVirtual, PointType>,
                                                PointType> {
    public:
        using parent_type =
                details::position_locator_base<position_locator<Deref, IsTransposed, IsVirtual, PointType>,
                                               position_iterator<Deref, IsTransposed, IsVirtual, PointType>,
                                               position_iterator<Deref, IsTransposed, IsVirtual, PointType>, PointType>;
        using deref_type = Deref;
        using const_locator = position_locator<typename Deref::const_adaptor, IsTransposed, IsVirtual, PointType>;
        using point_type = typename parent_type::point_type;
        using coordinate_type = typename parent_type::coordinate_type;
        using x_coordinate_type = typename parent_type::x_coordinate_type;
        using y_coordinate_type = typename parent_type::y_coordinate_type;
        using x_iterator = typename parent_type::x_iterator;
        using y_iterator = typename parent_type::y_iterator;
        using transposed_type = position_locator<Deref, not IsTransposed, IsVirtual, PointType>;
        using size_type = std::size_t;

        static constexpr bool is_transposed = IsTransposed;
        static constexpr bool is_mutable = true;

        template<typename NewDeref>
            requires IsPixelDereferenceAdaptor<NewDeref>
        struct add_deref {
            using type = position_locator<deref_compose<NewDeref, Deref>, IsTransposed, IsVirtual, PointType>;

            static constexpr auto make(position_locator const &it, NewDeref const &new_deref) -> type {
                return type(it.pos(), it.step(), deref_compose<NewDeref, Deref>{});
            }
        };

        explicit constexpr position_locator(PointType const &pos = {0, 0}, PointType const &step = {1, 1},
                                            Deref const &deref = {}, x_coordinate_type const width = 0,
                                            y_coordinate_type const height = 0,
                                            typename parent_type::value_type *storage = nullptr) :
            y_pos_(pos, step, deref, width, height, storage) {
        }
        constexpr position_locator(position_locator const &that) : y_pos_(that.y_pos_) {
        }
        constexpr auto operator=(position_locator const &that) -> position_locator & = default;

        constexpr ~position_locator() noexcept = default;

        template<typename Deref_, bool Transposed_, bool IsVirtual_, typename PointType_>
        explicit constexpr position_locator(position_locator<Deref_, Transposed_, IsVirtual_, PointType_> const &that) :
            y_pos_(that.y_pos_) {
        }
        template<typename Deref_, bool Transposed_, bool IsVirtual_, typename PointType_>
        explicit constexpr position_locator(position_locator<Deref_, Transposed_, IsVirtual_, PointType_> const &that,
                                            coordinate_type y_step) :
            y_pos_(that.pos(), point_type(that.step().x(), that.step().y() * y_step), that.deref()) {
        }
        template<typename Deref_, bool Transposed_, bool IsVirtual_, typename PointType_>
        explicit constexpr position_locator(position_locator<Deref_, Transposed_, IsVirtual_, PointType_> const &that,
                                            coordinate_type x_step, coordinate_type y_step, bool transpose = false) :
            y_pos_(that.pos(),
                   transpose ? point_type(that.step().x() * y_step, that.step.y() * x_step)
                             : point_type(that.step().x() * x_step, that.step.y() * y_step),
                   that.deref()) {
        }

        constexpr auto operator==(position_locator const &that) const -> bool {
            return y_pos_ == that.y_pos_;
        }
        constexpr auto operator!=(position_locator const &that) const -> bool {
            return y_pos_ != that.y_pos_;
        }

        constexpr auto x() -> x_iterator & {
            return *details::ptr_reinterpret_cast<x_iterator *>(this);
        }
        constexpr auto x() const -> x_iterator const & {
            return *details::ptr_reinterpret_cast_const<x_iterator const *>(this);
        }
        constexpr auto y() -> y_iterator & {
            return y_pos_;
        }
        constexpr auto y() const -> y_iterator const & {
            return y_pos_;
        }

        constexpr auto pos() const -> point_type const & {
            return y_pos_.pos();
        }
        constexpr auto pos() -> point_type & {
            return y_pos_.pos();
        }
        constexpr auto step() const -> point_type const & {
            return y_pos_.step();
        }
        constexpr auto deref() const -> deref_type const & {
            return y_pos_.deref();
        }

        constexpr auto width() -> x_coordinate_type & {
            return y_pos_.width();
        }
        constexpr auto height() -> y_coordinate_type & {
            return y_pos_.height();
        }
        [[nodiscard]] constexpr auto width() const -> x_coordinate_type const & {
            return y_pos_.width();
        }
        [[nodiscard]] constexpr auto height() const -> y_coordinate_type const & {
            return y_pos_.height();
        }

        constexpr auto is1DTraversable(x_coordinate_type) const -> bool {
            return false;
        }

        constexpr auto yDistanceTo(position_locator const &that, x_coordinate_type) const -> y_coordinate_type {
            if constexpr (IsTransposed) {
                return (that.pos().y() - pos().y()) / step().y();
            }
            return (that.pos().x() - pos().x()) / step().x();
        }

    private:
        y_iterator y_pos_;
    };

#ifndef MGIL_NO_COMPILE_TIME_TESTING
    static_assert(IsPixelLocator<position_locator<test_deref_adaptor>>);
    static_assert(IsPixelLocatorHasTransposedType<position_locator<test_deref_adaptor>>);

    static_assert(IsPixelIterator<two_dimensional_iterator<position_locator<test_deref_adaptor>>>);
#endif
} // namespace mgil

// The ImageView class
namespace mgil {
    template<typename Loc>
        requires IsPixelLocator<Loc>
    class ImageView {
    public:
        using value_type = typename Loc::value_type;
        using reference = typename Loc::reference;
        using coordinate_type = typename Loc::coordinate_type;
        using difference_type = coordinate_type;
        using const_view = ImageView<typename Loc::const_locator>;
        using point_type = typename Loc::point_type;
        using locator = Loc;
        using iterator = two_dimensional_iterator<Loc>;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using size_type = std::size_t;
        using xy_locator = locator;
        using x_iterator = typename xy_locator::x_iterator;
        using y_iterator = typename xy_locator::y_iterator;
        using x_coordinate_type = typename xy_locator::x_coordinate_type;
        using y_coordinate_type = typename xy_locator::y_coordinate_type;

        template<typename Deref>
            requires IsPixelDereferenceAdaptor<Deref>
        struct add_deref {
            using type = ImageView<typename Loc::template add_deref<Deref>::type>;
            static auto make(ImageView const &view, Deref const &deref) -> type {
                return type{Loc::template add_deref<Deref>::make(view.pixels(), deref)};
            }
        };

    private:
        x_coordinate_type width_;
        y_coordinate_type height_;
        xy_locator pixels_;

    public:
        constexpr ImageView() = default;
        template<typename View>
        explicit constexpr ImageView(View const &that) :
            width_(that.width_), height_(that.height_), pixels_(that.pixels_) {
        }
        template<typename Loc_>
            requires IsPixelLocator<Loc_>
        constexpr ImageView(point_type const &dims, Loc_ const &loc) :
            width_(dims.x()), height_(dims.y()), pixels_(loc) {
        }
        template<typename Loc_>
            requires IsPixelLocator<Loc_>
        constexpr ImageView(x_coordinate_type width, x_coordinate_type height, Loc_ const &loc) :
            width_(width), height_(height), pixels_(loc) {
        }

        template<typename View>
        constexpr auto operator=(View const &that) -> ImageView & {
            width_ = that.width();
            height_ = that.height();
            pixels_ = that.pixels();
            return *this;
        }
        constexpr auto operator=(ImageView const &that) -> ImageView & = default;

        template<typename View>
        constexpr auto operator==(View const &that) const -> bool {
            return width_ == that.width() and height_ == that.height() and pixels_ == that.pixels();
        }
        template<typename View>
        constexpr auto operator!=(View const &that) const -> bool {
            return !(*this == that);
        }

        constexpr auto width() const -> x_coordinate_type const & {
            return width_;
        }
        constexpr auto height() const -> y_coordinate_type const & {
            return height_;
        }
        constexpr auto pixels() const -> xy_locator const & {
            return pixels_;
        }
        constexpr auto width() -> x_coordinate_type & {
            return width_;
        }
        constexpr auto height() -> y_coordinate_type & {
            return height_;
        }
        constexpr auto pixels() -> xy_locator & {
            return pixels_;
        }

        [[nodiscard]] constexpr auto is1DTraversable() const -> bool {
            return pixels_.is1DTraversable(width_);
        }

        [[nodiscard]] constexpr auto empty() const -> bool {
            return not(width_ > 0 and height_ > 0);
        }
        [[nodiscard]] constexpr auto size() const -> size_type {
            return width_ * height_;
        }
        [[nodiscard]] constexpr auto begin() const -> iterator {
            return iterator(pixels_, width_, height_, 0, 0);
        }
        [[nodiscard]] constexpr auto end() const -> iterator {
            return iterator(pixels_, width_, height_, 0, height_);
        }
        [[nodiscard]] constexpr auto rbegin() const -> reverse_iterator {
            return reverse_iterator(end());
        }
        [[nodiscard]] constexpr auto rend() const -> reverse_iterator {
            return reverse_iterator(begin());
        }
        [[nodiscard]] constexpr auto front() const -> reference {
            return *begin();
        }
        [[nodiscard]] constexpr auto back() const -> reference {
            return *(end() - 1);
        }

        [[nodiscard]] constexpr auto operator[](difference_type i) const -> reference {
            assert(i < size());
            return begin()[i];
        }
        [[nodiscard]] constexpr auto at(difference_type i) const -> iterator {
            assert(i < size());
            return begin() + i;
        }
        [[nodiscard]] constexpr auto at(point_type const &i) const -> iterator {
            return at(i.x(), i.y());
        }
        [[nodiscard]] constexpr auto at(x_coordinate_type x, y_coordinate_type y) const -> iterator {
            assert(x >= 0 and x < width());
            assert(y >= 0 and y < height());
            return begin() + y * width() + x;
        }

        [[nodiscard]] constexpr auto operator()(point_type const &p) const -> reference {
            return operator()(p.x(), p.y());
        }
        [[nodiscard]] constexpr auto operator()(x_coordinate_type x, y_coordinate_type y) const -> reference {
            assert(x >= 0 and x < width());
            assert(y >= 0 and y < height());
            return pixels_(x, y);
        }

        [[nodiscard]] constexpr auto xyAt(x_coordinate_type x, y_coordinate_type y) const -> xy_locator {
            assert(x >= 0 and x < width());
            assert(y >= 0 and y < height());
            return pixels_ + point_type{x, y};
        }
        [[nodiscard]] constexpr auto xyAt(point_type const &p) const -> xy_locator {
            return xyAt(p.x(), p.y());
        }

        [[nodiscard]] constexpr auto xAt(x_coordinate_type x, y_coordinate_type y) const -> x_iterator {
            assert(x >= 0 and x < width());
            assert(y >= 0 and y < height());
            return pixels_.xAt(x, y);
        }
        [[nodiscard]] constexpr auto xAt(point_type const &p) const -> x_iterator {
            return xAt(p.x(), p.y());
        }
        [[nodiscard]] constexpr auto rowBegin(y_coordinate_type y) const -> x_iterator {
            return xAt(0, y);
        }
        [[nodiscard]] constexpr auto rowEnd(y_coordinate_type y) const -> x_iterator {
            return xAt(width(), y);
        }

        [[nodiscard]] constexpr auto yAt(x_coordinate_type x, y_coordinate_type y) const -> y_iterator {
            assert(x >= 0 and x < width());
            assert(y >= 0 and y < height());
            return pixels_.yAt(x, y);
        }
        [[nodiscard]] constexpr auto yAt(point_type const &p) const -> y_iterator {
            return yAt(p.x(), p.y());
        }
        [[nodiscard]] constexpr auto colBegin(x_coordinate_type x) const -> y_iterator {
            return yAt(x, 0);
        }
        [[nodiscard]] constexpr auto colEnd(x_coordinate_type x) const -> y_iterator {
            return yAt(x, height());
        }

        friend auto operator<<(std::ostream &os, ImageView view) -> std::ostream & {
            std::size_t counter = 0;
            for (auto const &pixel: view) {
                ++counter;
                if (counter == view.width()) {
                    os << pixel << "\n";
                    counter = 0;
                } else {
                    os << pixel << " ";
                }
            }
            return os;
        }
    };

#ifndef MGIL_NO_COMPILE_TIME_TESTING
    static_assert(IsImageView<ImageView<position_locator<test_deref_adaptor>>>);
#endif
} // namespace mgil

template<typename Loc>
struct std::formatter<mgil::ImageView<Loc>> {
    constexpr auto parse(std::format_parse_context &ctx) {
        return ctx.begin();
    }

    constexpr auto format(mgil::ImageView<Loc> const &view, std::format_context &ctx) const {
        std::string temp;
        std::size_t counter = 0;
        std::format_to(std::back_inserter(temp), "ImageView(width = {}, height = {}):\n", view.width(), view.height());
        for (auto const &pixel: view) {
            ++counter;
            if (counter == view.width()) {
                std::format_to(std::back_inserter(temp), "{}\n", pixel);
                counter = 0;
            } else {
                std::format_to(std::back_inserter(temp), "{} ", pixel);
            }
        }
        return std::format_to(ctx.out(), "{}", temp);
    }
};

// Image view factories
namespace mgil {
    // Produce a width*height view of zero-initialized pixels
    template<typename Pixel>
        requires IsPixel<Pixel>
    class EmptyView {
        struct EmptyDeref : deref_base<EmptyDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            constexpr auto operator()(Point<std::ptrdiff_t> const &) const -> Pixel {
                return Pixel{0};
            }
        };
        using locator = position_locator<EmptyDeref>;
        using point_type = Point<std::int64_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(std::ptrdiff_t width, std::ptrdiff_t height) const -> view_type {
            assert(width >= 0);
            assert(height >= 0);
            return view_type(width, height, locator());
        }
    };

    template<typename Pixel>
        requires IsPixel<Pixel>
    constexpr auto empty(std::ptrdiff_t width, std::ptrdiff_t height) {
        return EmptyView<Pixel>{}(width, height);
    }

    // Produce a width*height view of designated pixel
    template<typename Pixel>
        requires IsPixel<Pixel>
    class IdenticalView {
        struct IdenticalDeref : deref_base<IdenticalDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            Pixel pixel;
            constexpr auto operator()(Point<std::ptrdiff_t> const &) const -> Pixel {
                return pixel;
            }
        };
        using locator = position_locator<IdenticalDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(std::ptrdiff_t width, std::ptrdiff_t height, Pixel const &pixel) const -> view_type {
            assert(width >= 0);
            assert(height >= 0);
            return view_type(width, height, locator({0, 0}, {1, 1}, IdenticalDeref{.pixel = pixel}));
        }
    };

    template<typename Pixel>
        requires IsPixel<Pixel>
    constexpr auto identical(std::ptrdiff_t width, std::ptrdiff_t height, Pixel const &pixel) {
        return IdenticalView<Pixel>{}(width, height, pixel);
    }

    // Produce a width*height view of a gradient pixel(x,y) = start + x*step_x + y*step_y
    template<typename Pixel>
        requires IsPixel<Pixel>
    class GradientView {
        struct GradientDeref : deref_base<GradientDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            Pixel start;
            Pixel step_x;
            Pixel step_y;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                return start + pos.x() * step_x + pos.y() * step_y;
            }
        };
        using locator = position_locator<GradientDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(std::ptrdiff_t width, std::ptrdiff_t height, Pixel const &start, Pixel const &step_x,
                                  Pixel const &step_y) const -> view_type {
            assert(width >= 0);
            assert(height >= 0);
            return view_type(
                    width, height,
                    locator({0, 0}, {1, 1}, GradientDeref{.start = start, .step_x = step_x, .step_y = step_y}));
        }
    };

    template<typename Pixel>
        requires IsPixel<Pixel>
    constexpr auto gradient(std::ptrdiff_t width, std::ptrdiff_t height, Pixel const &start, Pixel const &step_x,
                            Pixel const &step_y) {
        return GradientView<Pixel>{}(width, height, start, step_x, step_y);
    }

    // Call gen(x, y) to produce a pixel value for each coordinate
    template<typename Pixel>
        requires IsPixel<Pixel>
    class GenerateView {
        struct GenerateDeref : deref_base<GenerateDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            std::function<Pixel(std::ptrdiff_t, std::ptrdiff_t)> func;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                return func(pos.x(), pos.y());
            }
        };
        using locator = position_locator<GenerateDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(std::ptrdiff_t width, std::ptrdiff_t height, auto const &generator) const
                -> view_type {
            assert(width >= 0);
            assert(height >= 0);
            return view_type(width, height, locator({0, 0}, {1, 1}, GenerateDeref{.func = generator}));
        }
    };

    template<typename Gen, typename Pixel = std::invoke_result_t<Gen, std::ptrdiff_t, std::ptrdiff_t>>
        requires IsPixel<Pixel>
    constexpr auto generate(std::ptrdiff_t width, std::ptrdiff_t height, Gen generator) {
        return GenerateView<Pixel>{}(width, height, generator);
    }

    // Tiles a h*w pattern across a larger rows*cols image
    template<typename Pixel, typename View>
        requires IsPixel<Pixel> and IsImageView<View>
    class PatternView {
        struct PatternDeref : deref_base<PatternDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            View view;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                return view(pos.x() % view.width(), pos.y() % view.height());
            }
        };
        using locator = position_locator<PatternDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View view, std::ptrdiff_t width, std::ptrdiff_t height) const -> view_type {
            assert(view.width() <= width);
            assert(view.height() <= height);
            return view_type(width, height, locator({0, 0}, {1, 1}, PatternDeref{.view = view}));
        }
    };

    template<typename View, typename Pixel = typename View::value_type>
        requires IsPixel<Pixel> and IsImageView<View>
    constexpr auto pattern(View view, std::ptrdiff_t width, std::ptrdiff_t height) {
        return PatternView<Pixel, View>{}(view, width, height);
    }

    // Concat two views of equal height side-by-side
    template<typename Pixel, typename View1, typename View2>
        requires IsPixel<Pixel> and IsImageView<View1> and IsImageView<View2> and IsImageViewsCompatible<View1, View2>
    class ConcatHorizontalView {
        struct ConcatHorizontalDeref : deref_base<ConcatHorizontalDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            View1 view1;
            View2 view2;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                if (pos.x() < view1.width()) {
                    return view1(pos.x(), pos.y());
                }
                if (pos.x() >= view1.width() and pos.x() < view1.width() + view2.width()) {
                    return view2(pos.x() - view1.width(), pos.y());
                }
                std::unreachable();
            }
        };
        using locator = position_locator<ConcatHorizontalDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View1 view1, View2 view2) const {
            assert(view1.height() == view2.height());
            return view_type(view1.width() + view2.width(), view1.height(),
                             locator({0, 0}, {1, 1}, ConcatHorizontalDeref{.view1 = view1, .view2 = view2}));
        }
    };

    template<typename View1, typename View2, typename Pixel = typename View1::value_type>
        requires IsPixel<Pixel> and IsImageView<View1> and IsImageView<View2> and IsImageViewsCompatible<View1, View2>
    constexpr auto concatHorizontal(View1 view1, View2 view2) {
        return ConcatHorizontalView<Pixel, View1, View2>{}(view1, view2);
    }

    // Concat two views of equal height side-by-side
    template<typename Pixel, typename View1, typename View2>
        requires IsPixel<Pixel> and IsImageView<View1> and IsImageView<View2> and IsImageViewsCompatible<View1, View2>
    class ConcatVerticalView {
        struct ConcatVerticalDeref : deref_base<ConcatVerticalDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            View1 view1;
            View2 view2;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                if (pos.y() < view1.height()) {
                    return view1(pos.x(), pos.y());
                }
                if (pos.y() >= view1.height() and pos.x() < view1.height() + view2.height()) {
                    return view2(pos.x(), pos.y() - view1.height());
                }
                std::unreachable();
            }
        };
        using locator = position_locator<ConcatVerticalDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View1 view1, View2 view2) const {
            assert(view1.width() == view2.width());
            return view_type(view1.width(), view1.height() + view2.height(),
                             locator({0, 0}, {1, 1}, ConcatVerticalDeref{.view1 = view1, .view2 = view2}));
        }
    };

    template<typename View1, typename View2, typename Pixel = typename View1::value_type>
        requires IsPixel<Pixel> and IsImageView<View1> and IsImageView<View2> and IsImageViewsCompatible<View1, View2>
    constexpr auto concatVertical(View1 view1, View2 view2) {
        return ConcatVerticalView<Pixel, View1, View2>{}(view1, view2);
    }

    // Creates a two-color checkerboard with cell size cw*ch
    template<typename Pixel>
        requires IsPixel<Pixel>
    class CheckerboardView {
        struct CheckerboardDeref : deref_base<CheckerboardDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            Pixel pixel1;
            Pixel pixel2;
            std::ptrdiff_t cell_width;
            std::ptrdiff_t cell_height;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                auto const x_relative = pos.x() / cell_width;
                auto const y_relative = pos.y() / cell_height;
                if ((x_relative % 2 == 0 and y_relative % 2 == 0) or (x_relative % 2 == 1 and y_relative % 2 == 1)) {
                    return pixel1;
                }
                return pixel2;
            }
        };
        using locator = position_locator<CheckerboardDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(std::ptrdiff_t width, std::ptrdiff_t height, Pixel const &pixel1, Pixel const &pixel2,
                                  std::ptrdiff_t cell_width, std::ptrdiff_t cell_height) const {
            assert(width >= 0);
            assert(height >= 0);
            assert(cell_width <= width and cell_width > 0);
            assert(cell_height <= height and cell_height > 0);
            return view_type(width, height,
                             locator({0, 0}, {1, 1},
                                     CheckerboardDeref{.pixel1 = pixel1,
                                                       .pixel2 = pixel2,
                                                       .cell_width = cell_width,
                                                       .cell_height = cell_height}));
        }
    };

    template<typename Pixel>
        requires IsPixel<Pixel>
    constexpr auto checkerboard(std::ptrdiff_t width, std::ptrdiff_t height, Pixel const &pixel1, Pixel const &pixel2,
                                std::ptrdiff_t cell_width, std::ptrdiff_t cell_height) {
        return CheckerboardView<Pixel>{}(width, height, pixel1, pixel2, cell_width, cell_height);
    }

    // Fills an image of size width*height with uniform random pixels using the provided pseudo random generator rng
    template<typename Pixel, typename Gen>
        requires IsPixel<Pixel>
    class NoiseUniformView {
        struct NoiseUniformDeref : deref_base<NoiseUniformDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            std::random_device rd;
            Gen gen{rd()};
            using channel_type = typename Pixel::channel_type;
            channel_type max{};
            channel_type min{};
            constexpr auto getDistribution() const {
                if constexpr (std::is_integral_v<channel_type>) {
                    return std::uniform_int_distribution<channel_type>(min, max);
                } else if constexpr (std::is_floating_point_v<channel_type>) {
                    return std::uniform_real_distribution<channel_type>(min, max);
                }
            }
            std::conditional_t<std::is_integral_v<channel_type>, std::uniform_int_distribution<channel_type>,
                               std::uniform_real_distribution<channel_type>>
                    distrib = getDistribution();
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                std::array<channel_type, Pixel::getSize()> arr;
                for (std::size_t i = 0; i < Pixel::getSize(); ++i) {
                    // Warning for potential UB
                    arr[i] = const_cast<NoiseUniformDeref *>(this)->distrib(const_cast<NoiseUniformDeref *>(this)->gen);
                }
                return Pixel(arr);
            }

            constexpr NoiseUniformDeref() = default;
            constexpr NoiseUniformDeref(channel_type max, channel_type min) : max(max), min(min) {};
            constexpr NoiseUniformDeref(NoiseUniformDeref const &that) : max(that.max), min(that.min) {};
        };
        using locator = position_locator<NoiseUniformDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;
        using channel_type = typename Pixel::channel_type;

    public:
        constexpr auto operator()(std::ptrdiff_t width, std::ptrdiff_t height, channel_type min,
                                  channel_type max) const {
            assert(width >= 0);
            assert(height >= 0);
            assert(max >= min);
            return view_type(width, height, locator({0, 0}, {1, 1}, NoiseUniformDeref(max, min)));
        }
    };

    template<typename Pixel, typename Gen>
        requires IsPixel<Pixel>
    constexpr auto noiseUniform(std::ptrdiff_t width, std::ptrdiff_t height, typename Pixel::channel_type min,
                                typename Pixel::channel_type max) {
        return NoiseUniformView<Pixel, Gen>{}(width, height, min, max);
    }

    template<typename Range, typename Pixel = details::infer_std_range_over_pixel_t<Range>>
        requires IsPixel<Pixel> and
                 (std::ranges::sized_range<Range> or
                  (std::ranges::forward_range<Range> and std::ranges::range<std::ranges::range_value_t<Range>>))
    class FromRangeView {
        struct FromRangeDeref : deref_base<FromRangeDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            std::ranges::iterator_t<Range> begin{};
            std::ranges::iterator_t<Range> end{};
            std::ptrdiff_t width{};
            std::ptrdiff_t height{};
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel
                requires(not std::ranges::range<std::ranges::range_value_t<Range>>)
            {
                return *(begin + (pos.y() * width + pos.x()));
            }
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                auto row_begin = (begin + pos.y())->begin();
                return *(row_begin + pos.x());
            }
        };
        using locator = position_locator<FromRangeDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(Range &&pixels, std::ptrdiff_t width, std::ptrdiff_t height)
            requires(not std::ranges::range<std::ranges::range_value_t<Range>>)
        {
            assert(width >= 0);
            assert(height >= 0);
            assert(std::ranges::size(pixels) == width * height);
            return view_type(width, height,
                             locator({0, 0}, {1, 1},
                                     FromRangeDeref{.begin = std::ranges::begin(std::forward<Range>(pixels)),
                                                    .end = std::ranges::end(std::forward<Range>(pixels)),
                                                    .width = width,
                                                    .height = height}));
        }
        constexpr auto operator()(Range &&pixels)
            requires(std::ranges::range<std::ranges::range_value_t<Range>>)
        {
            std::ptrdiff_t height = std::ranges::size(pixels);
            std::ptrdiff_t width = std::ranges::size(*std::ranges::begin(pixels));
            return view_type(width, height,
                             locator({0, 0}, {1, 1},
                                     FromRangeDeref{.begin = std::ranges::begin(std::forward<Range>(pixels)),
                                                    .end = std::ranges::end(std::forward<Range>(pixels)),
                                                    .width = width,
                                                    .height = height}));
        }
    };

    template<typename Range, typename Pixel = details::infer_std_range_over_pixel_t<std::remove_cvref_t<Range>>>
    constexpr auto fromRange(Range &&pixels, std::ptrdiff_t width, std::ptrdiff_t height) {
        return FromRangeView<Range, Pixel>{}(std::forward<Range>(pixels), width, height);
    }
    template<typename Range, typename Pixel = details::infer_std_range_over_pixel_t<std::remove_cvref_t<Range>>>
    constexpr auto fromRange(Range &&pixels) {
        return FromRangeView<Range, Pixel>{}(std::forward<Range>(pixels));
    }
} // namespace mgil

// Image view adaptors
namespace mgil {
    // Lazily projects a rectangular ROI
    template<typename Pixel, typename View>
        requires IsPixel<Pixel> and IsImageView<View>
    class CropView {
        struct CropDeref : deref_base<CropDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            View view;
            std::ptrdiff_t x;
            std::ptrdiff_t y;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                auto original_x = pos.x() + x;
                auto original_y = pos.y() + y;
                return view(original_x, original_y);
            }
        };
        using locator = position_locator<CropDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View view, std::ptrdiff_t x, std::ptrdiff_t y, std::ptrdiff_t width,
                                  std::ptrdiff_t height) {
            assert(width >= 0);
            assert(height >= 0);
            assert(x + width <= view.width());
            assert(y + height <= view.height());
            return view_type(width, height, locator({0, 0}, {1, 1}, CropDeref{.view = view, .x = x, .y = y}));
        }
    };

    struct CropFn : details::image_adaptor_closure_tag {
        template<typename View, typename Pixel = typename View::value_type>
        constexpr auto operator()(View view, std::ptrdiff_t x, std::ptrdiff_t y, std::ptrdiff_t width,
                                  std::ptrdiff_t height) const {
            return CropView<Pixel, View>{}(view, x, y, width, height);
        }

        constexpr auto operator()(std::ptrdiff_t x, std::ptrdiff_t y, std::ptrdiff_t width,
                                  std::ptrdiff_t height) const {
            return details::pipeable{[this, x, y, width, height]<typename View>(View view) {
                return (*this)(view, x, y, width, height);
            }};
        }
    };

    inline constexpr auto crop = CropFn{};

    // Flip the image horizontally
    template<typename Pixel, typename View>
        requires IsPixel<Pixel> and IsImageView<View>
    class FlipHorizontalView {
        struct FlipHorizontalDeref : deref_base<FlipHorizontalDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            View view;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                return view(view.width() - pos.x() - 1, pos.y());
            }
        };
        using locator = position_locator<FlipHorizontalDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View view) {
            return view_type(view.width(), view.height(), locator({0, 0}, {1, 1}, FlipHorizontalDeref{.view = view}));
        }
    };

    struct FlipHorizontalFn : details::image_adaptor_closure_tag {
        template<typename View, typename Pixel = typename View::value_type>
        constexpr auto operator()(View view) const {
            return FlipHorizontalView<Pixel, View>{}(view);
        }

        constexpr auto operator()() const {
            return details::pipeable{[this]<typename View>(View view) { return (*this)(view); }};
        }
    };

    inline constexpr auto flipHorizontal = FlipHorizontalFn{};

    // Flip the image vertically
    template<typename Pixel, typename View>
        requires IsPixel<Pixel> and IsImageView<View>
    class FlipVerticalView {
        struct FlipVerticalDeref : deref_base<FlipVerticalDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            View view;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                return view(pos.x(), view.height() - pos.y() - 1);
            }
        };
        using locator = position_locator<FlipVerticalDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View view) {
            return view_type(view.width(), view.height(), locator({0, 0}, {1, 1}, FlipVerticalDeref{.view = view}));
        }
    };

    struct FlipVerticalFn : details::image_adaptor_closure_tag {
        template<typename View, typename Pixel = typename View::value_type>
        constexpr auto operator()(View view) const {
            return FlipVerticalView<Pixel, View>{}(view);
        }

        constexpr auto operator()() const {
            return details::pipeable{[this]<typename View>(View view) { return (*this)(view); }};
        }
    };

    inline constexpr auto flipVertical = FlipVerticalFn{};

    // Rotate the image by 90 degrees clockwise
    template<typename Pixel, typename View>
        requires IsPixel<Pixel> and IsImageView<View>
    class Rotate90View {
        struct Rotate90Deref : deref_base<Rotate90Deref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            View view;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                return view(pos.y(), view.height() - pos.x() - 1);
            }
        };
        using locator = position_locator<Rotate90Deref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View view) {
            return view_type(view.height(), view.width(), locator({0, 0}, {1, 1}, Rotate90Deref{.view = view}));
        }
    };

    struct Rotate90Fn : details::image_adaptor_closure_tag {
        template<typename View, typename Pixel = typename View::value_type>
        constexpr auto operator()(View view) const {
            return Rotate90View<Pixel, View>{}(view);
        }

        constexpr auto operator()() const {
            return details::pipeable{[this]<typename View>(View view) { return (*this)(view); }};
        }
    };

    inline constexpr auto rotate90 = Rotate90Fn{};

    // Rotate the image by 180 degrees clockwise
    template<typename Pixel, typename View>
        requires IsPixel<Pixel> and IsImageView<View>
    class Rotate180View {
        struct Rotate180Deref : deref_base<Rotate180Deref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            View view;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                return view(view.width() - pos.x() - 1, view.height() - pos.y() - 1);
            }
        };
        using locator = position_locator<Rotate180Deref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View view) {
            return view_type(view.width(), view.height(), locator({0, 0}, {1, 1}, Rotate180Deref{.view = view}));
        }
    };

    struct Rotate180Fn : details::image_adaptor_closure_tag {
        template<typename View, typename Pixel = typename View::value_type>
        constexpr auto operator()(View view) const {
            return Rotate180View<Pixel, View>{}(view);
        }

        constexpr auto operator()() const {
            return details::pipeable{[this]<typename View>(View view) { return (*this)(view); }};
        }
    };

    inline constexpr auto rotate180 = Rotate180Fn{};

    // Rotate the image by 180 degrees clockwise
    template<typename Pixel, typename View>
        requires IsPixel<Pixel> and IsImageView<View>
    class Rotate270View {
        struct Rotate270Deref : deref_base<Rotate270Deref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            View view;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                return view(view.width() - pos.y() - 1, pos.x());
            }
        };
        using locator = position_locator<Rotate270Deref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View view) {
            return view_type(view.height(), view.width(), locator({0, 0}, {1, 1}, Rotate270Deref{.view = view}));
        }
    };

    struct Rotate270Fn : details::image_adaptor_closure_tag {
        template<typename View, typename Pixel = typename View::value_type>
        constexpr auto operator()(View view) const {
            return Rotate270View<Pixel, View>{}(view);
        }

        constexpr auto operator()() const {
            return details::pipeable{[this]<typename View>(View view) { return (*this)(view); }};
        }
    };

    inline constexpr auto rotate270 = Rotate270Fn{};

    // Extracts a single channel as a grayscale view
    template<typename Pixel_, typename View>
        requires IsPixel<Pixel_> and IsImageView<View>
    class ChannelExtractByIndexView {
    protected:
        using channel_type = typename Pixel_::channel_type;
        using result_pixel_type = Pixel<channel_type, gray_layout_t>;
        struct ChannelExtractDeref : deref_base<ChannelExtractDeref, result_pixel_type, result_pixel_type &,
                                                result_pixel_type const &, result_pixel_type, false> {
            View view;
            std::size_t index;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> result_pixel_type {
                return result_pixel_type(view(pos.x(), pos.y()).get(index));
            }
        };
        using locator = position_locator<ChannelExtractDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View view, std::size_t const index) {
            return view_type(view.height(), view.width(),
                             locator({0, 0}, {1, 1}, ChannelExtractDeref{.view = view, .index = index}));
        }
    };

    struct ChannelExtractByIndexFn : details::image_adaptor_closure_tag {
        template<typename View, typename Pixel = typename View::value_type>
        constexpr auto operator()(View view, std::size_t const index) const {
            return ChannelExtractByIndexView<Pixel, View>{}(view, index);
        }

        constexpr auto operator()(std::size_t const index) const {
            return details::pipeable{[this, index]<typename View>(View view) { return (*this)(view, index); }};
        }
    };

    inline constexpr auto channelExtractByIndex = ChannelExtractByIndexFn{};

    // Transform pixels by calling a function on them
    template<typename Pixel, typename View>
        requires IsPixel<Pixel> and IsImageView<View>
    class TransformView {
        struct TransformDeref : deref_base<TransformDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            std::function<Pixel(Pixel const &)> func;
            View view;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                return func(view(pos.x(), pos.y()));
            }
        };
        using locator = position_locator<TransformDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View view, auto const &func) const -> view_type {
            return view_type(view.width(), view.height(),
                             locator({0, 0}, {1, 1}, TransformDeref{.func = func, .view = view}));
        }
    };

    struct TransformFn : details::image_adaptor_closure_tag {
        template<typename View, typename Pixel = typename View::value_type>
        constexpr auto operator()(View view, auto const &func) const {
            return TransformView<Pixel, View>{}(view, func);
        }

        constexpr auto operator()(auto const &func) const {
            return details::pipeable{[this, func]<typename View>(View view) { return (*this)(view, func); }};
        }
    };

    inline constexpr auto transform = TransformFn{};

    // Convert color space layouts
    template<typename SrcPixel, typename DstPixel, typename View>
        requires IsPixel<SrcPixel> and IsPixel<DstPixel> and IsImageView<View> and
                 IsPixelsColorConvertible<SrcPixel, DstPixel>
    class ColorConvertView {
        struct ColorConvertDeref
            : deref_base<ColorConvertDeref, DstPixel, DstPixel &, DstPixel const &, DstPixel, false> {
            View view;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> DstPixel {
                return PixelTraits<SrcPixel>::template convertTo<DstPixel>(view(pos.x(), pos.y()));
            }
        };
        using locator = position_locator<ColorConvertDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View view) const -> view_type {
            return view_type(view.width(), view.height(), locator({0, 0}, {1, 1}, ColorConvertDeref{.view = view}));
        }
    };

    struct ColorConvertFn : details::image_adaptor_closure_tag {
        template<typename DstType, typename DstLayout, typename DstPixel = Pixel<DstType, DstLayout>, typename View,
                 typename SrcPixel = typename View::value_type>
        constexpr auto operator()(View view, DstType const &type_tag, DstLayout const &layout_tag) const {
            return ColorConvertView<SrcPixel, DstPixel, View>{}(view);
        }

        template<typename DstType, typename DstLayout>
        constexpr auto operator()(DstType const &type_tag, DstLayout const &layout_tag) const {
            return details::pipeable{[this, type_tag, layout_tag]<typename View>(View view) {
                return (*this)(view, type_tag, layout_tag);
            }};
        }
    };

    inline constexpr auto colorConvert = ColorConvertFn{};

    // Subsample the image using the nearest neighbor interpolation
    template<typename Pixel, typename View>
        requires IsPixel<Pixel> and IsImageView<View>
    class NearestView {
        struct NearestDeref : deref_base<NearestDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            View view;
            std::ptrdiff_t result_width;
            std::ptrdiff_t result_height;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                double const x_scale = static_cast<double>(view.width()) / static_cast<double>(result_width);
                double const y_scale = static_cast<double>(view.height()) / static_cast<double>(result_height);
                double const scaled_x = static_cast<double>(pos.x()) * x_scale;
                double const scaled_y = static_cast<double>(pos.y()) * y_scale;
                auto rounded_x = static_cast<std::ptrdiff_t>(round(scaled_x));
                auto rounded_y = static_cast<std::ptrdiff_t>(round(scaled_y));
                if (rounded_x >= view.width()) {
                    rounded_x = view.width() - 1;
                }
                if (rounded_y >= view.height()) {
                    rounded_y = view.height() - 1;
                }
                return view(rounded_x, rounded_y);
            }
        };
        using locator = position_locator<NearestDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View view, std::ptrdiff_t result_width, std::ptrdiff_t result_height) const {
            assert(width >= 0);
            assert(height >= 0);
            assert(result_width >= 0);
            assert(result_height >= 0);
            return view_type(
                    result_width, result_height,
                    locator({0, 0}, {1, 1},
                            NearestDeref{.view = view, .result_width = result_width, .result_height = result_height}));
        }
    };

    struct NearestFn : details::image_adaptor_closure_tag {
        template<typename View, typename Pixel = typename View::value_type>
        constexpr auto operator()(View view, std::ptrdiff_t result_width, std::ptrdiff_t result_height) const {
            return NearestView<Pixel, View>{}(view, result_width, result_height);
        }

        constexpr auto operator()(std::ptrdiff_t result_width, std::ptrdiff_t result_height) const {
            return details::pipeable{[this, result_width, result_height]<typename View>(View view) {
                return (*this)(view, result_width, result_height);
            }};
        }
    };

    inline constexpr auto nearest = NearestFn{};

    // Pad the image with given constant
    template<typename Pixel, typename View>
        requires IsPixel<Pixel> and IsImageView<View>
    class PadConstantView {
        struct PadConstantDeref : deref_base<PadConstantDeref, Pixel, Pixel &, Pixel const &, Pixel, false> {
            View view;
            std::ptrdiff_t pad_x;
            std::ptrdiff_t pad_y;
            Pixel pad_pixel;
            constexpr auto operator()(Point<std::ptrdiff_t> const &pos) const -> Pixel {
                assert(pos.x() >= 0 and pos.x() < view.width() + pad_x * 2);
                assert(pos.y() >= 0 and pos.y() < view.height() + pad_y * 2);
                auto origin_x = pos.x() - pad_x;
                auto origin_y = pos.y() - pad_y;
                if (origin_x >= 0 and origin_x < view.width() and origin_y >= 0 and origin_y < view.height()) {
                    return view(origin_x, origin_y);
                }
                return pad_pixel;
            }
        };
        using locator = position_locator<PadConstantDeref>;
        using point_type = Point<std::ptrdiff_t>;
        using view_type = ImageView<locator>;

    public:
        constexpr auto operator()(View view, std::ptrdiff_t pad_x, std::ptrdiff_t pad_y, Pixel const &pad_pixel) const {
            assert(pad_x >= 0);
            assert(pad_y >= 0);
            return view_type(
                    view.width() + pad_x * 2, view.height() + pad_y * 2,
                    locator({0, 0}, {1, 1},
                            PadConstantDeref{.view = view, .pad_x = pad_x, .pad_y = pad_y, .pad_pixel = pad_pixel}));
        }
    };

    struct PadConstantFn : details::image_adaptor_closure_tag {
        template<typename View, typename Pixel = typename View::value_type>
        constexpr auto operator()(View view, std::ptrdiff_t pad_x, std::ptrdiff_t pad_y, Pixel const &pad_pixel) const {
            return PadConstantView<Pixel, View>{}(view, pad_x, pad_y, pad_pixel);
        }

        template<typename Pixel>
        constexpr auto operator()(std::ptrdiff_t pad_x, std::ptrdiff_t pad_y, Pixel const &pad_pixel) const {
            return details::pipeable{[this, pad_x, pad_y, pad_pixel]<typename View>(View view) {
                return (*this)(view, pad_x, pad_y, pad_pixel);
            }};
        }
    };

    inline constexpr auto padConstant = PadConstantFn{};
} // namespace mgil

// Image container
namespace mgil {
    // An owning container of image consists of a two-dimensional map of pixels
    // Has deep copying semantics, may cause massive performance regression when being abused
    // Always prefer non-owning image views instead of image containers.
    template<typename Pixel, typename Alloc = std::allocator<Pixel>>
        requires IsPixel<Pixel>
    class Image {
    public:
        using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<Pixel>;
        using locator = position_locator<identity_deref_adaptor<Pixel>, false, false>;
        using view_type = ImageView<locator>;
        using const_view_type = ImageView<typename locator::const_locator>;
        using point_type = typename view_type::point_type;
        using value_type = typename view_type::value_type;
        using x_coordinate_type = typename view_type::x_coordinate_type;
        using y_coordinate_type = typename view_type::y_coordinate_type;

    private:
        allocator_type allocator_;
        Pixel *data_;
        x_coordinate_type width_;
        y_coordinate_type height_;
        std::mdspan<Pixel, std::dextents<std::size_t, 2>> interface_;

        constexpr auto deallocate() -> void {
            if (data_ != nullptr) {
                allocator_.deallocate(data_);
            }
        }

    public:
        constexpr Image() : allocator_(), data_(nullptr), width_(), height_(), interface_(data_, width_, height_) {
        }
        // Deep copy
        constexpr Image(Image const &that) : allocator_(), width_(that.width_), height_(that.height_) {
            data_ = allocator_.allocate(width_ * height_);
            std::ranges::copy(that.data_, data_);
            interface_ = std::mdspan<Pixel, std::dextents<std::size_t, 2>>{data_, width_, height_};
        }
        // Shallow copy
        constexpr Image(Image &&that) noexcept :
            allocator_(), data_(std::move(that.data_)), width_(that.width_), height_(that.height_),
            interface_(std::move(that.interface_)) {
            that.data_ = nullptr;
        }
        constexpr auto operator=(Image const &that) -> Image & {
            if (this == std::addressof(that)) {
                return *this;
            }
            deallocate();
            width_ = that.width_;
            height_ = that.height_;
            data_ = allocator_.allocate(width_ * height_);
            std::ranges::copy(that.data_, data_);
            interface_ = std::mdspan<Pixel, std::dextents<std::size_t, 2>>{data_, width_, height_};
            return *this;
        }
        constexpr auto operator=(Image &&that) noexcept -> Image & {
            if (this == std::addressof(that)) {
                return *this;
            }
            deallocate();
            data_ = that.data_;
            width_ = that.width_;
            height_ = that.height_;
            interface_ = std::move(that.interface_);
            that.data_ = nullptr;
            return *this;
        }

        constexpr Image(x_coordinate_type width, y_coordinate_type height, Pixel const &initial = {},
                        allocator_type const alloc = {}) : allocator_(alloc), width_(width), height_(height) {
            data_ = allocator_.allocate(width_ * height_);
            for (std::size_t i = 0; i < width_ * height_; ++i) {
                data_[i] = initial;
            }
            interface_ = std::mdspan<Pixel, std::dextents<std::size_t, 2>>{data_, width_, height_};
        }
        template<typename Pixel_, typename Alloc_>
            requires IsPixelsCompatible<Pixel_, Pixel>
        explicit constexpr Image(Image<Pixel_, Alloc_> const &that) : width_(that.width_), height_(that.height_) {
            data_ = allocator_.allocate(width_ * height_);
            for (std::size_t i = 0; i < width_ * height_; ++i) {
                data_[i] = Pixel{that.data_[i]};
            }
            interface_ = std::mdspan<Pixel, std::dextents<std::size_t, 2>>{data_, width_, height_};
        }
        template<typename Pixel_, typename Alloc_>
            requires IsPixelsCompatible<Pixel_, Pixel>
        constexpr auto operator=(Image<Pixel_, Alloc_> const &that) -> Image & {
            if (this == std::addressof(that)) {
                return *this;
            }
            deallocate();
            width_ = that.width_;
            height_ = that.height_;
            data_ = allocator_.allocate(width_ * height_);
            for (std::size_t i = 0; i < width_ * height_; ++i) {
                data_[i] = Pixel{that.data_[i]};
            }
            interface_ = std::mdspan<Pixel, std::dextents<std::size_t, 2>>{data_, width_, height_};
            return *this;
        }

        template<typename View>
            requires IsImageView<std::remove_cvref_t<View>>
        explicit constexpr Image(View &&view) :
            width_(std::forward<View>(view).width()), height_(std::forward<View>(view).height()) {
            data_ = allocator_.allocate(width_ * height_);
            interface_ = std::mdspan<Pixel, std::dextents<std::size_t, 2>>{data_, width_, height_};
            for (std::size_t x = 0; x < width_; ++x) {
                for (std::size_t y = 0; y < height_; ++y) {
                    interface_[x, y] = std::forward<View>(view)(x, y);
                }
            }
        }
        template<typename View>
            requires IsImageView<std::remove_cvref_t<View>>
        constexpr auto operator=(View &&view) -> Image & {
            deallocate();
            width_ = std::forward<View>(view).width();
            height_ = std::forward<View>(view).height();
            data_ = allocator_.allocate(width_ * height_);
            interface_ = std::mdspan<Pixel, std::dextents<std::size_t, 2>>{data_, width_, height_};
            for (std::size_t x = 0; x < width_; ++x) {
                for (std::size_t y = 0; y < height_; ++y) {
                    interface_[x, y] = std::forward<View>(view)(x, y);
                }
            }
            return *this;
        }

        constexpr auto operator==(Image const &that) const noexcept -> bool {
            return data_ == that.data_ and width_ == that.width_ && height_ == that.height_;
        }

        constexpr auto operator[](x_coordinate_type x, y_coordinate_type y) const -> Pixel const & {
            return interface_[x, y];
        }
        constexpr auto operator[](x_coordinate_type x, y_coordinate_type y) -> Pixel & {
            return interface_[x, y];
        }
        constexpr auto operator[](point_type const &pos) const -> Pixel const & {
            return interface_[pos.x(), pos.y()];
        }
        constexpr auto operator[](point_type const &pos) -> Pixel & {
            return interface_[pos.x(), pos.y()];
        }

        [[nodiscard]] constexpr auto toView() const -> view_type {
            return view_type(width_, height_, locator({0, 0}, {1, 1}, {}, width_, height_, data_));
        }
    };

#ifndef MGIL_NO_COMPILE_TIME_TESTING
    static_assert(IsImageContainer<Image<Pixel<int, rgb_layout_t>>>);
#endif
} // namespace mgil

// Image processing algorithms
namespace mgil {
    // convolve
    template<typename View1, typename View2, typename Image = Image<typename View1::value_type>>
        requires IsImageView<View1> and IsImageView<View2> and IsImageContainer<Image>
    constexpr auto convolve(View1 src, View2 kernel) -> Image {
        std::size_t const kw = kernel.width();
        std::size_t const kh = kernel.height();
        std::size_t const w = src.width();
        std::size_t const h = src.height();
        std::size_t const padx = kw / 2;
        std::size_t const pady = kh / 2;
        assert(kw <= w);
        assert(kh <= h);

        Image image(w, h);

        using pixel_type = typename View1::value_type;
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                Pixel<float, typename View1::value_type::layout_type> sum{};
                for (std::size_t j = 0; j < kh; ++j) {
                    for (std::size_t i = 0; i < kw; ++i) {
                        std::size_t ix = std::clamp<int>(int(x) + int(i) - int(padx), 0, w - 1);
                        std::size_t iy = std::clamp<int>(int(y) + int(j) - int(pady), 0, h - 1);
                        auto converted_src = src(ix, iy).template castTo<float>();
                        auto converted_kernel = kernel(i, j).template castTo<float>();
                        sum += converted_src * converted_kernel;
                    }
                }
                image[x, y] = sum.template castTo<typename View1::value_type::channel_type>();
            }
        }

        return std::move(image);
    }
    // boxBlur
    template<typename View, typename Image = Image<typename View::value_type>>
        requires IsImageView<View> and IsImageContainer<Image>
    constexpr auto boxBlur(View src, std::size_t const kw, std::size_t const kh) -> Image {
        using pixel = Pixel<float, typename View::value_type::layout_type>;
        pixel kernel_pixel(1.0f / (kw * kh));
        std::vector data(kw * kh, kernel_pixel);
        auto kernel = fromRange(data, kw, kh);
        return std::move(convolve(src, kernel));
    }
    // gaussianBlur
    template<typename View, typename Image = Image<typename View::value_type>>
        requires IsImageView<View> and IsImageContainer<Image>
    constexpr auto gaussianBlur(View src, float const sigma) -> Image {
        int const radius = static_cast<int>(std::ceil(3 * sigma));
        int const size = 2 * radius + 1;
        using pixel = Pixel<float, typename View::value_type::layout_type>;
        std::vector<pixel> data(size * size);
        float sum = 0.0f;
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                float val = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
                pixel kernel_pixel(val);
                data[(y + radius) * size + (x + radius)] = kernel_pixel;
                sum += val;
            }
        }
        for (auto &elem: data) {
            elem /= sum;
        }
        auto kernel = fromRange(data, size, size);
        return std::move(convolve(src, kernel));
    }
    // erode, dilate, open, and close
    template<typename View1, typename View2, typename Comparator, typename Image = Image<typename View1::value_type>>
        requires IsImageView<View1> and IsImageContainer<Image> and IsImageView<View2> and
                 std::same_as<typename View2::value_type::channel_type, bool>
    constexpr auto morphologicalOperation(View1 src, View2 se, Comparator cmp) -> Image {
        auto const w = src.width();
        auto const h = src.height();
        std::size_t const kw = se.width();
        std::size_t const kh = se.height();
        auto const padx = kw / 2;
        auto const pady = kh / 2;
        Image image(w, h);
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) {
                bool first = true;
                typename View1::value_type result;
                for (std::size_t j = 0; j < kh; ++j) {
                    for (std::size_t i = 0; i < kw; ++i) {
                        if (not se(i, j)) {
                            continue;
                        }
                        std::size_t const ix = std::clamp<int>(int(x) + int(i) + int(padx), 0, w - 1);
                        std::size_t const iy = std::clamp<int>(int(y) + int(j) + int(pady), 0, h - 1);
                        auto val = src(ix, iy);
                        if (first or cmp(val, result)) {
                            result = val;
                            first = false;
                        }
                    }
                }
                image[x, y] = result;
            }
        }
        return std::move(image);
    }
    inline constexpr auto erode = [](auto src, auto se) {
        return std::move(morphologicalOperation(src, se, [](auto a, auto b) { return a < b; }));
    };
    inline constexpr auto dilate = [](auto src, auto se) {
        return std::move(morphologicalOperation(src, se, [](auto a, auto b) { return a > b; }));
    };
    template<typename View1, typename View2, typename Comparator, typename Image = Image<typename View1::value_type>>
        requires IsImageView<View1> and IsImageContainer<Image> and IsImageView<View2> and
                 std::same_as<typename View2::value_type::channel_type, bool>
    constexpr auto open(View1 src, View2 se) {
        Image temp = erode(src, se);
        return std::move(dilate(temp.toView(), se));
    }
    template<typename View1, typename View2, typename Comparator, typename Image = Image<typename View1::value_type>>
        requires IsImageView<View1> and IsImageContainer<Image> and IsImageView<View2> and
                 std::same_as<typename View2::value_type::channel_type, bool>
    constexpr auto close(View1 src, View2 se) {
        Image temp = dilate(src, se);
        return std::move(erode(temp.toView(), se));
    }
    // sobel
    template<typename View, typename Image = Image<typename View::value_type>>
        requires IsImageView<View> and IsImageContainer<Image>
    constexpr auto sobel(View src) -> Image {
        using pixel = Pixel<float, typename View::value_type::layout_type>;
        constexpr pixel kx[9] = {pixel{-1}, pixel{0},  pixel{1}, pixel{-2}, pixel{0},
                                 pixel{2},  pixel{-1}, pixel{0}, pixel{1}};
        constexpr pixel ky[9] = {pixel{-1}, pixel{-2}, pixel{-1}, pixel{0}, pixel{0},
                                 pixel{0},  pixel{1},  pixel{2},  pixel{1}};
        auto kernel1 = fromRange(kx, 3, 3);
        auto kernel2 = fromRange(ky, 3, 3);
        auto image1 = convolve(src, kernel1);
        auto image2 = convolve(src, kernel2);
        Image result(src.width(), src.height());
        for (std::size_t y = 0; y < src.height(); ++y) {
            for (std::size_t x = 0; x < src.width(); ++x) {
                for (std::size_t i = 0; i < pixel::getSize(); ++i) {
                    result[x, y].get(i) = std::hypot(static_cast<float>(image1[x, y].get(i)),
                                                     static_cast<float>(image2[x, y].get(i)));
                }
            }
        }
        return std::move(result);
    }
} // namespace mgil

namespace mgil::inline concepts {
    template<typename T>
    concept IsImageFileIOClass = requires {
        typename T::value_type;
        typename T::image_type;

        requires IsPixel<typename T::value_type>;
        requires IsImageContainer<typename T::image_type>;

        T::readFile(std::declval<std::filesystem::path>());
        T::writeFile(std::declval<typename T::image_type::view_type>(), std::declval<std::filesystem::path>());
    };
} // namespace mgil::inline concepts

// BMP Image file I/O
namespace mgil {
    // A small helper that automatically calls fclose on scope exit
    struct FileRAIIHelper {
        FILE *fp;
        ~FileRAIIHelper() {
            if (not fp) {
                fclose(fp);
            }
        }
    };

    // A simple BMP image file reader/writer, ported from previous project
    // So far it can only handle the file store in uncompressed 24-bit BGR format
    // But extending it should be easy
    template<typename Pixel = Pixel<UInt8_0255, rgb_layout_t>, typename Image = Image<Pixel>>
    struct BMPFileIO {
        using value_type = Pixel;
        using image_type = Image;
#pragma pack(push, 1)
        struct BMPFileHeader {
            std::uint16_t type;
            std::uint32_t size;
            std::uint16_t reserved1;
            std::uint16_t reserved2;
            std::uint32_t offset;
        };
        static_assert(sizeof(BMPFileHeader) == 14, "BMPFileHeader size mismatch");

        struct BMPInfoHeader {
            std::uint32_t size;
            std::int32_t width;
            std::int32_t height;
            std::uint16_t planes;
            std::uint16_t bits_count;
            std::uint32_t compression;
            std::uint32_t image_size;
            std::int32_t x_pixels_per_m;
            std::int32_t y_pixels_per_m;
            std::uint32_t colors_used;
            std::uint32_t colors_important;
        };
        static_assert(sizeof(BMPInfoHeader) == 40, "BMPInfoHeader size mismatch");
#pragma pack(pop)

        enum class BMPFileIOError {
            FILE_NOT_FOUND,
            NOT_A_FILE,
            COULD_NOT_OPEN_FILE,
            FILE_READ_FAILED,
            FILE_WRITE_FAILED,
            HEADER_FORMAT_ERROR,
            FILE_FORMAT_ERROR,
            INFO_HEADER_ERROR,
            UNSUPPORTED
        };

        static_assert(IsLayoutCompatible<typename Pixel::layout_type, rgb_layout_t>,
                      "Only 24-bit RGB layout compatible formats are supported");

        static auto readFile(std::filesystem::path const &image_path) -> std::expected<Image, BMPFileIOError> {
            namespace fs = std::filesystem;

            if (not fs::exists(image_path)) {
                return std::unexpected{BMPFileIOError::FILE_NOT_FOUND};
            }
            if (fs::is_directory(image_path)) {
                return std::unexpected{BMPFileIOError::NOT_A_FILE};
            }

            FileRAIIHelper file{.fp = fopen(image_path.string().c_str(), "rb")};
            BMPFileHeader header;
            BMPInfoHeader info_header;
            if (not file.fp) {
                return std::unexpected{BMPFileIOError::COULD_NOT_OPEN_FILE};
            }

            if (fread(&header, sizeof(BMPFileHeader), 1, file.fp) != 1) {
                return std::unexpected{BMPFileIOError::FILE_READ_FAILED};
            }
            if (header.type != 0x4D42) {
                return std::unexpected{BMPFileIOError::HEADER_FORMAT_ERROR};
            }
            if (fread(&info_header, sizeof(BMPInfoHeader), 1, file.fp) != 1) {
                return std::unexpected{BMPFileIOError::FILE_READ_FAILED};
            }
            if (info_header.bits_count != 24 or info_header.compression != 0) {
                return std::unexpected{BMPFileIOError::UNSUPPORTED};
            }

            auto width = info_header.width, height = std::abs(info_header.height);
            Image image(width, height);

            fseek(file.fp, header.offset, SEEK_SET);

            for (std::size_t y = 0; y < height; ++y) {
                for (std::size_t x = 0; x < width; ++x) {
                    std::uint8_t raw_pixel[3];
                    if (fread(&raw_pixel, 3, 1, file.fp) != 1) {
                        return std::unexpected{BMPFileIOError::FILE_READ_FAILED};
                    }
                    Pixel pixel;
                    pixel.template get<red_color_t>() = raw_pixel[2];
                    pixel.template get<green_color_t>() = raw_pixel[1];
                    pixel.template get<blue_color_t>() = raw_pixel[0];
                    image[x, y] = pixel;
                }
            }

            return image;
        }

        template<typename View>
            requires IsImageView<View>
        static auto writeFile(View view, std::filesystem::path const &image_path)
                -> std::expected<void, BMPFileIOError> {
            static_assert(IsLayoutCompatible<typename View::value_type::layout_type, rgb_layout_t>,
                          "Only 24-bit RGB layout compatible formats are supported");
            namespace fs = std::filesystem;

            if (fs::is_directory(image_path)) {
                return std::unexpected{BMPFileIOError::NOT_A_FILE};
            }

            FileRAIIHelper file{.fp = fopen(image_path.string().c_str(), "wb")};
            if (not file.fp) {
                return std::unexpected{BMPFileIOError::COULD_NOT_OPEN_FILE};
            }

            std::int32_t const width = view.width();
            std::int32_t const height = view.height();
            std::uint32_t const file_size = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + view.size();

            BMPFileHeader const header = {
                    .type = 0x4D42, .size = file_size, .offset = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader)};
            BMPInfoHeader const info_header = {.size = sizeof(BMPInfoHeader),
                                               .width = width,
                                               .height = height,
                                               .planes = 1,
                                               .bits_count = 24,
                                               .compression = 0,
                                               .image_size = file_size,
                                               .x_pixels_per_m = 0,
                                               .y_pixels_per_m = 0};

            if (fwrite(&header, sizeof(BMPFileHeader), 1, file.fp) != 1 or
                fwrite(&info_header, sizeof(BMPInfoHeader), 1, file.fp) != 1) {
                return std::unexpected{BMPFileIOError::FILE_WRITE_FAILED};
            }

            for (std::size_t y = 0; y < height; ++y) {
                for (std::size_t x = 0; x < width; ++x) {
                    std::uint8_t raw_pixel[3];
                    raw_pixel[2] = view(x, y).template get<red_color_t>();
                    raw_pixel[1] = view(x, y).template get<green_color_t>();
                    raw_pixel[0] = view(x, y).template get<blue_color_t>();
                    if (fwrite(&raw_pixel, 3, 1, file.fp) != 1) {
                        return std::unexpected{BMPFileIOError::FILE_WRITE_FAILED};
                    }
                }
            }

            return {};
        }
    };

    template<typename FileIO>
        requires IsImageFileIOClass<FileIO>
    constexpr auto readImage(std::filesystem::path const &path) {
        return FileIO::readFile(path);
    }
    template<typename FileIO, typename View>
        requires IsImageView<View> and IsImageFileIOClass<FileIO>
    constexpr auto writeImage(View view, std::filesystem::path const &path) {
        return FileIO::writeFile(view, path);
    }

#ifndef MGIL_NO_COMPILE_TIME_TESTING
    static_assert(IsImageFileIOClass<BMPFileIO<>>);
#endif
} // namespace mgil

#endif // MGIL_H
