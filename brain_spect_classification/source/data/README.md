### データセット作成手順

1. brain_spect_classification\source\data\get_file_list.py

- 全データのリストを作成

2. brain_spect_classification\source\data\get_type_file_list.py

- 画像タイプ別にリストを作成

3. brain_spect_classification\source\data\generate_data_group_list.py

- クロスバリデーションのためのリストを作成

4. グラフを作成する

- csvを読み込み

- グループnを取り出し

- 1datファイルを開く(root pathを読み込み)

- ポイントをサンプリング

- グラフを作成

出力は，以下のようになっている
(pickleで保存)

`grouped_graph_n`

|||
|:---:|:---:|
|case 1|(number of sampling point) * (number of sampling point)|
|case 2| ... |

`grouped_intensity_n`

|||
|:---:|:---:|
|case 1| intensity (length = number of sampling point) |
|case 2| ... |

`grouped_label_n`

|||
|:---:|:---:|
|case 1| label |
|case 2| ... |
