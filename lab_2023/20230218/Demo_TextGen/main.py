import gradio as gr
from transformers import T5Tokenizer, AutoModelForCausalLM, GenerationConfig

# 0. モデルとトークナイザーの定義
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-small")
tokenizer.do_lower_case = True  # rinna/japanese-gpt2特有のハック
model = AutoModelForCausalLM.from_pretrained(
    "rinna/japanese-gpt2-small",
    pad_token_id=tokenizer.eos_token_id  # warningを避けるために、padにEOSトークンを割りあてる
    )

# 1. Gradioのコンポーネントのイベント処理用の関数の定義
def generate(text, max_length, num_beams, p):
    """初回のテキスト生成

    テキスト生成を行うが、デコード方法によって異なる結果になることを示すための処理を行う。
    指定されたパラメタを使って、異なる４つデコード方法を同時に出力する。

    Args:
        str: Stateから取得（続きを生成するためのプロンプト）
        int: Sliderから取得（全てのデコード方法に共通のパラメタ。生成する単語数）
        int: Sliderから取得（Beam Searchのパラメタ）
        int: Sliderから取得（Top-p Samplingのパラメタ）
    
    Returns:
        str: State（生成結果を入出力の状態に反映）
        str: TextArea（全文表示用のコンポーネントで使用）
        str: TextArea（今回生成した文を表示するコンポーネントで使用）
    """
    # テキスト生成用のconfigクラスを使って、４パターンの設定を定義する。
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

        # 読みやすくさの処理を行なって、リストに追加
        generated_texts.append("。\n".join(generated.replace(" ", "").split("。")))

    # gradioはtupleを想定している。
    # これと同じ処理：return generated_texts[0], generated_texts[1], generated_texts[2]
    # pythonのタプルは「,」によって生成される。丸括弧は省略可能。
    # 参考：https://note.nkmk.me/python-function-return-multiple-values/
    return tuple(generated_texts)

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

# 2. GradioによるUI/イベント処理の定義
with gr.Blocks() as demo:
    # 2.1. UI
    gr.Markdown('''
    # テキスト生成
    テキストを入力すると、４パターンのデコード方法でテキスト生成を実行します。
    ## ４つのパターン（入門編）
    1. Greedy: ビームサーチもサンプリングも行いません。毎回、最も確率の高い単語を選択します。
    2. Sampling: モデルによって与えられた語彙全体の確率分布に基づいて次の単語を選択します。
    3. Beam Search: 各タイムステップで複数の仮説を保持し、最終的に仮説ごとのシーケンス全体で最も高い確率を持つ仮説を選択します。
    4. Top-p Sampling: 2の方法に関して、確率の和がpになる最小の単語にフィルタリングすることで、確率が低い単語が選ばれる可能性を無くします。
    ''')
    with gr.Row():
        with gr.Column():
            input_text = ...  # テキストを入力するコンポーネント
            max_length = ...  # テキストを入力するコンポーネント
            num_beams = ...  # テキストを入力するコンポーネント
            p = ...    # テキストを入力するコンポーネント
            btn = gr.Button("Decode")
        
        with gr.Column():
            out1 = ...  # Greedy decode output
            out2 = ...  # Smapling decode output
            out3 = ...  # Beam Search decode output
            out4 = ...  # Top-p Sampling decode output

    with gr.Row():
        pass  # レイアウト（gr.Column()の使い方）
        pass  # 初回の結果としてout1を表示させる
        pass  # どの結果を使うか、radioで選択できる

        pass  # 生成した続きを、これまでの結果を含めて、表示する
        pass  # どのデコード方法で、続きを生成するか、radioで選択する  
        pass  # 生成した続きを、これまでの結果を含めずに、表示する
    
    # 2.2 イベント処理
    btn.click(fn=generate, inputs=[input_text, max_length, num_beams, p], outputs=[out1, out2, out3, out4])

if __name__ == "__main__":
    demo.launch()
