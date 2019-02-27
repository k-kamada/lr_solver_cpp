# 概説
- 単純な線形回帰（Linear Regression）問題を解くためのプログラムです。
- アルゴリズムとして最小二乗法(MSE)、Lasso回帰、Ridge回帰を選択することができます。

# プロジェクト構成
- lr_solver.hpp : ライブラリ本体
- sample.cpp : 上記ライブラリの使用例
- Boston.csv : sample.cppで用いるサンプルデータ
- README.md : 本ファイル

# 依存ライブラリについて
- Eigen : 公式から入手したヘッダファイルを配置しパスを通すか、パッケージマネージャ等でインストール可能

# ビルド環境
- gcc 8.2.0
- Eigen 3.3.5.1

# 入力ファイル形式
- 特定の文字をデリミタとするCSVファイルである必要があります。
- データは数値である必要があります（アルファベット混在、文字のみ、といった項は処理できません）。

# 動作手順
- lr_solver.hppをインクルードし、LR__Solverクラスのインスタンスを作成します。この際、読み込むCSVファイルの名前（相対パス）と、サンプルIDを示すデータ列、目的変数を示すデータ列のインデックスを指定してください。
- 作成したインスタンスに対し、アルゴリズム毎のメソッドを実行するとEigenのベクトル形式で推定された係数を得ることができます。
- 得た係数ベクトルをshow_results()メソッドに渡すと、対応する説明変数のラベルと共に結果を表示できます。
