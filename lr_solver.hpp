#ifndef INCLUDE_LRSOLVER
#define INCLUDE_LRSOLVER

#include <cstdint>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <eigen3/Eigen/Dense>

class LR_Solver {
    public:
        // コンストラクタにファイル名とデリミタ、サンプルID列番号と目的変数列番号を渡す
        LR_Solver(const char *input_filename, uint32_t id_index, uint32_t y_index, bool intercept=true, char delimiter=',') 
            :id_index(id_index), 
            y_index(y_index), 
            intercept(intercept),
            delimiter(delimiter) 
        {
            std::vector<std::string> temp_data_matrix; 
            std::ifstream ifs(input_filename);
            if (ifs) {
                std::string line;
                uint32_t line_cnt = 0;
                while(std::getline(ifs, line)) {
                    if (line_cnt == 0) {
                        // 各説明変数の名称を取得、vectorに保持
                        column_names = split_to_strvec(line, delimiter);
                        p = column_names.size() - 2;
                    } else {
                        // データ本体を一行ずつstringのvectorに保持
                        temp_data_matrix.push_back(line);
                    }
                    ++line_cnt;
                }
                N = line_cnt - 1;
                // stringのvectorからデータ本体を生成
                data = gen_eigen_matrix(temp_data_matrix, N, p);
            } else {
                std::cout << "Failed to open file: " << input_filename << std::endl;
            }
        }
        // 最小二乗法によるbetaの算出
        Eigen::VectorXf solve_mse() {
            // Xの用意
            int M;
            if (intercept) 
                M = p+1;
            else 
                M = p;
            Eigen::MatrixXf X(N, M);
            auto copied_data = data;
            standardize_all(copied_data);
            if (intercept)
                X << Eigen::VectorXf::Ones(N), copied_data;
            else
                X << copied_data;
            // 係数ベクトル
            Eigen::VectorXf beta(M);
            if (N < p) { // 本当は各列が相互に線形独立かどうかの判定も必要だが簡単のため
                std::cout << "Error: X is not a full rank Matrix" << std::endl;
                return beta;
            }
            Eigen::MatrixXf Xt = X.transpose();
            beta = (Xt*X).inverse() * Xt * Y;
            return beta;
        }
        // Ridge回帰によるbetaの算出
        Eigen::VectorXf solve_ridge(float lambda) {
            // Xの用意
            int M;
            if (intercept) 
                M = p+1;
            else 
                M = p;
            Eigen::MatrixXf X(N, M);
            auto copied_data = data;
            // 列ごとに標準化
            standardize_all(copied_data);
            if (intercept)
                X << Eigen::VectorXf::Ones(N), copied_data;
            else
                X << copied_data;
            // 係数ベクトル
            Eigen::VectorXf beta(M);
            Eigen::MatrixXf Xt = X.transpose();
            auto I = Eigen::MatrixXf::Identity(M, M);
            beta = (Xt*X + lambda*I).inverse() * Xt * Y;
            return beta;
        }
        // ADMM(Alternating Direction Method of Multipliers)を採用したLasso回帰によるbetaの算出(is_coef_beta = falseの時はthetaを返す)
        Eigen::VectorXf solve_lasso_admm(uint32_t max_iter, float rho, float lambda, bool is_coef_beta=true) {
            // Xの用意
            int M;
            if (intercept) 
                M = p+1;
            else 
                M = p;
            Eigen::MatrixXf X(N, M);
            Eigen::MatrixXf copied_data = data;
            // 列ごとに標準化
            standardize_all(copied_data);
            if (intercept)
                X << Eigen::VectorXf::Ones(N), copied_data;
            else
                X << copied_data;
            Eigen::MatrixXf Xt = X.transpose();
            Eigen::MatrixXf I = Eigen::MatrixXf::Identity(M, M);
            // 各種初期化
            Eigen::VectorXf beta(M);
            beta = Eigen::VectorXf::Zero(beta.size());
            Eigen::VectorXf theta(M);
            theta = Eigen::VectorXf::Zero(theta.size());
            Eigen::VectorXf mu(M);
            mu = Eigen::VectorXf::Zero(mu.size());
            // 変数を含まない項の事前計算
            Eigen::MatrixXf beta_coef_matrix = (Xt*X/N + rho*I).inverse();
            Eigen::MatrixXf Xt_Y_N = Xt * Y / N;

            for (uint32_t i=0; i<max_iter; i++) {
                beta = beta_coef_matrix * (Xt_Y_N + rho*theta - mu);
                theta = soft_threshold(lambda/rho, beta+(mu/rho));
                mu = mu + rho * (beta - theta);
            }

            if (is_coef_beta) {
                return beta;
            } else {
                return theta;
            }
        }
        // 解betaを渡すと、対応する説明変数と共に係数を一覧表示
        void show_results(Eigen::VectorXf &beta) {
            uint32_t fieldname_index = 0;
            // 浮動小数点数の表示桁数を固定
            std::cout << std::fixed;
            for (auto i=0; i<beta.size(); ++i) {
                // サンプルIDまたは目的変数は結果表示に使わないため、スキップ
                while (fieldname_index == id_index || fieldname_index == y_index) {
                    fieldname_index++;
                    if (fieldname_index > p+2) {
                        std::cout << "Something wrong!" << std::endl;
                        return;
                    }
                }
                if (i==0 && intercept) {
                    // beta0項は元データにないため、こちらで名前をつける
                    std::cout << std::setw(12) << std::setprecision(10) << "\"bias\"" << "\t" << std::showpos << std::right << beta(i) << std::endl; 
                } else {
                    std::cout << std::setw(12) << std::setprecision(10) << column_names[fieldname_index] << "\t" << std::showpos << std::right << beta(i) << std::endl;
                    fieldname_index++;
                }
            }
            // このまま抜けると大域に影響してしまうので書式リセット
            std::cout << std::noshowpos;
            std::cout << std::setw(0);
            std::cout << std::setprecision(6);
            std::cout << std::defaultfloat;
        }
    private:
        // フィッティングに用いる行列X, ベクトルYを作成
        Eigen::MatrixXf gen_eigen_matrix(std::vector<std::string> &str_vector, int32_t N, int32_t p) {
            Eigen::MatrixXf data(N, p);
            Eigen::VectorXf temp_y(N);
            std::string field;
            float temp_val;
            uint32_t row_count = 0;
            uint32_t col_count = 0;
            uint32_t x_count = 0;
            uint32_t y_count = 0;

            for (auto &&e : str_vector) {
                std::istringstream stream(e);
                col_count = 0;
                x_count = 0;
                while (std::getline(stream, field, delimiter)) {
                    if (x_count == id_index) { // サンプルID列は説明変数に含めず、読み飛ばす
                        x_count++;
                        continue;
                    } else if (x_count == y_index) { // 目的変数の列のみ、temp_yに一旦格納してからYに渡す。説明変数には含めない
                        std::istringstream(field) >> temp_val; // string->float変換
                        temp_y(y_count) = temp_val;
                        x_count++;
                        y_count++;
                        continue;
                    }
                    std::istringstream(field) >> temp_val; // string->float変換
                    data(row_count, col_count) = temp_val;
                    col_count++;
                    x_count++;
                }
                row_count++;
            }
            Y = temp_y;
            return data;
        }
        // 列すべての標準化
        void standardize_all(Eigen::MatrixXf &X) {
            for (auto i=0; i<X.cols(); ++i) {
                standardize_column(X, i);
            }
        }
        // 列ごとの標準化（平均0, 分散1）
        void standardize_column(Eigen::MatrixXf &X, int index) {
            float mean = X.col(index).mean();
            float variance = 0.0;
            for (auto i=0; i<X.rows(); ++i) {
                variance += std::pow(X(i,index) - mean, 2.0);
            }
            variance /= X.rows();
            float std = std::sqrt(variance);

            for (auto i=0; i<X.rows(); ++i) {
                X(i,index) = (X(i,index) - mean) / std;  
            }
        }
        // thをスレッショルドとした軟判別閾値関数
        Eigen::VectorXf soft_threshold(float th, Eigen::VectorXf A) {
            Eigen::VectorXf result(A.size());
            for (auto i=0; i<A.size(); ++i) {
                if (A(i) > th) {
                    result(i) = A(i) - th;
                } else if (A(i) < -th) {
                    result(i) = A(i) + th;
                } else {
                    result(i) = 0.0;
                }
            }
            return result;
        }
        // stringを受け取り、デリミタで分割してvector<string>で返す
        std::vector<std::string> split_to_strvec(std::string &input, char delimiter) {
            std::istringstream stream(input);
            std::string field;
            std::vector<std::string> result;

            while (std::getline(stream, field, delimiter)) {
                result.push_back(field);
            }
            return result;
        }
        std::vector<std::string> column_names; // 説明変数の名前
        Eigen::MatrixXf data; // 説明変数の行列
        Eigen::VectorXf Y; // 目的変数のベクトル
        uint32_t N; // サンプル数
        uint32_t p; // 説明変数の数(ID列、目的変数列は除く)
        uint32_t id_index; // サンプルIDの列番号(0始まり)
        uint32_t y_index; // 目的変数の列番号（0始まり)
        bool intercept; // 切片項を含むかどうか
        char delimiter; // 元データの区切り文字
};

    
#endif
