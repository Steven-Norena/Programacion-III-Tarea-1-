//
// Created by STEVEN on 2/04/2026.
//

#ifndef TAREA_1_TENSOR_H
#define TAREA_1_TENSOR_H

#include <vector>

class TensorTransform;

class Tensor {
private:
    std::vector<size_t> shape;
    double* data;
    size_t total_size = 1;
public:
    Tensor(const std::vector<size_t>& shape,
        const std::vector<double>& values);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();
    [[nodiscard]] Tensor apply(const TensorTransform& transform) const;
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape) ;
    static Tensor random(const std::vector<size_t>& shape, double min, double max);
    static Tensor arange(double start, double end);
};

class TensorTransform {
public:
    [[nodiscard]] virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

#endif //TAREA_1_TENSOR_H