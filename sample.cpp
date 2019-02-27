#include "lr_solver.hpp"

int main() {
    // データセットのファイル名、データIDの列、目的変数の列、切片項を含めるかを指定
    LR_Solver dataset("Boston.csv", 0, 14, true);

    // Lasso(ADMM)
    int max_iter = 1000;
    float lambda_l = 1.0;
    float rho_l = 1.0;
    auto beta_l = dataset.solve_lasso_admm(max_iter, rho_l, lambda_l);
    std::cout << "Lasso ADMM:(lambda=" << lambda_l << ", rho=" << rho_l << ")" << std::endl;
    dataset.show_results(beta_l);
    
    std::cout << std::endl;

    // Ridge
    float lambda_r = 1.0;
    std::cout << "Ridge(lambda=" << lambda_r << ")" << std::endl;
    auto beta_r = dataset.solve_ridge(lambda_r);
    dataset.show_results(beta_r);

    std::cout << std::endl;

    // MSE
    std::cout << "MSE:" << std::endl;
    auto beta_m = dataset.solve_mse();
    dataset.show_results(beta_m);
}
