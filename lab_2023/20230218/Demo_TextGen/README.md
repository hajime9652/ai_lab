# デモ実装（テキスト生成）

## 環境構築
今回は必要なライブラリは三つです。Mecabは使用しません。
- transformers
- sentencepiece
- gradio

Windowsの場合
```
python3 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Macの場合
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```


# 課題
## その１（対象：全員）
４パターンのデコード方法の結果を同時に出力するデモを実装してください。

### テキスト生成の関数generation()の実装
main.pyの下記のコメントを参考に、必要な処理を実装してください。
コメントのある箇所は全て実装が必要な箇所です。
```python
def generate(text, max_length, num_beams, p):
    generate_config_list = [
        GenerationConfig(
            max_new_tokens=max_length,
            no_repeat_ngram_size=3,
            # Greedyの設定
        ),
        GenerationConfig(
            max_new_tokens=max_length,
            no_repeat_ngram_size=3,
            # Smaplingの設定
        ),
        GenerationConfig(
            max_new_tokens=max_length,
            no_repeat_ngram_size=3,
            # Beam Searchの設定
        ),
        GenerationConfig(
            max_new_tokens=max_length,
            no_repeat_ngram_size=3,
            # Top-p Smaplingの設定
        )
    ]
    generated_texts = []

    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    for generate_config in generate_config_list:
        output = ...  # modelを使ってテキスト生成を行います。

        generated = ...  # tokenizerを使って、outputを単語に変換します。
        generated_texts.append("。\n".join(generated.replace(" ", "").split("。")))
    return tuple(generated_texts)
```

### Gradioのコンポーネントの実装
main.pyの下記のコメントを参考に、必要な処理を実装してください。
コメントのある箇所は全て実装が必要な箇所です。
コンポーネントの引数は参考です。そのままで与えても動きません。
```python
with gr.Row():
    with gr.Column():
        input_text = ...  # テキストを入力するコンポーネント
        max_length = ...  # 数値を入力するコンポーネント.参考：（min=100, max=1000, step=100, default=100）
        num_beams = ...  # 数値を入力するコンポーネント.参考：（min=1, max=10, step=1, default=6）
        p = ...    # テキストを入力するコンポーネント.参考：（min=0, max=1, step=0.01, default=0.92）
        btn = gr.Button("Decode")
    
    with gr.Column():
        out1 = ...  # Greedy decode outputを表示するコンポーネント
        out2 = ...  # Smapling decode outputを表示するコンポーネント
        out3 = ...  # Beam Search decode outputを表示するコンポーネント
        out4 = ...  # Top-p Sampling decode outputを表示するコンポーネント
```

## その２（対象：その１が早めに終わった人）
４パターンのデコード方法の結果から、どれか一つ選んで、続きを出力するデモを実装してください。

以下の要求を満たすように実装してください。
- ４つのデコード結果から１つを選んで、初回のデコード結果を取得できる
- 初回のデコード結果を読み込んで、続きを生成する際に、４つのデコード方法から１つ選び、続きのテキスト生成が実行できる
- 以後、続きのテキストを入力して、実行するたびに、結果が増えていく
- 続きのテキスト生成を実行した際には、続きの部分を含めて全文を確認することができる（入力の確認）
- 続きのテキスト生成を実行した際には、続きの部分だけを確認することができる（出力の確認）


以下の条件を使って実装してください。
- `gr.State()`を使って、入出力の変数（状態）を管理する
    - 初期値はout1の結果とするために、out1が生成された時に入出力の変数数に代入する
    - 実行後は生成結果を入出力の変数数に代入し、次回実行時の入力とする
- 入出力のテキストの確認には、`gr.TextArea()`を使う
- イベント処理として以下の三つの関数を使う。`generate_next()`は処理の実装が必要
```python
def select_out1(out1):
    """out1が生成された時に、out1を後続の処理のデフォルト値に入力
    """
    return out1, out1, out1

def select_out(radio, out1, out2, out3, out4):
    """後続の処理に使用する、初回の処理結果を選択する
    """
    if radio == "1.Greedy":
        out = out1
    elif radio == "2.Sampling":
        out = out2
    elif radio == "3.Beam Search":
        out = out3
    else:
        out = out4
    return out, out, out

def generate_next(now_text, radio, max_length, num_beams, p):
    """続き生成

    これまで出力したテキストを入力して受け取り、続きを生成する。
    デコード方法を指定することができるが、そのパラメタは初回のテキスト生成と同じになる。

    Args:
        str: Stateから取得（続きを生成するためのプロンプト）
        str: Radioから取得（使用するデコード方法の名前）
        int: Sliderから取得（初回のテキスト生成で使用した値をここでも使用）
        int: Sliderから取得（初回のテキスト生成で使用した値をここでも使用）
        int: Sliderから取得（初回のテキスト生成で使用した値をここでも使用）
    
    Returns:
        str: State（生成結果を入出力の状態に反映）
        str: TextArea（全文表示用のコンポーネントで使用）
        str: TextArea（今回生成した文を表示するコンポーネントで使用）

    Todo:
        * 処理を実装
    """
    pass
```

