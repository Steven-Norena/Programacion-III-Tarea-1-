//
// Created by STEVEN on 6/04/2026.
//
#include "Tensor.cpp"


int main() {
        std::cout << "--- Iniciando simulacion de Red Neuronal ---" << std::endl;

        std::cout << "Paso 1: Creacion de Tensor" << std::endl;
        Tensor input = Tensor::random({1000, 20, 20}, -1.0, 1.0);

        std::cout << "Paso 2: Transformacion con view" << std::endl;
        Tensor input_flat = input.view({1000, 400});

        std::cout << "Paso 3: Multiplicacion de matriz" << std::endl;
        Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
        Tensor z1 = Matmul(input_flat, W1);

        std::cout << "Paso 4: Suma de matriz" << std::endl;
        Tensor b1 = Tensor::random({1, 100}, -0.1, 0.1);
        Tensor a1_pre = z1 + b1;

        std::cout << "Paso 5: Aplicacion de ReLU" << std::endl;
        ReLU relu;
        Tensor a1 = a1_pre.apply(relu);

        std::cout << "Paso 6: Multiplicacion de matriz" << std::endl;
        Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
        Tensor z2 = Matmul(a1, W2);

        std::cout << "Paso 7: Suma de matriz" << std::endl;
        Tensor b2 = Tensor::random({1, 10}, -0.1, 0.1);
        Tensor a2_pre = z2 + b2;

        std::cout << "Paso 8: Aplicacion de Sigmoid" << std::endl;
        Sigmoid sigmoid;
        Tensor output = a2_pre.apply(sigmoid);

    return 0;
}