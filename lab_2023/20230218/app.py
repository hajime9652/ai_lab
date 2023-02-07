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
            num_beams=1,  # beam幅の設定、２以上ではbeam searchになる。
            do_sample=False  # Samplingの設定
        ),
        GenerationConfig(
            max_new_tokens=max_length,
            no_repeat_ngram_size=3,
            num_beams=1,
            do_sample=True
        ),
        GenerationConfig(
            max_new_tokens=max_length,
            no_repeat_ngram_size=3,
            num_beams=num_beams,
            do_sample=False
        ),
        GenerationConfig(
            max_new_tokens=max_length,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_p=p  # Top-p Samplingのパラメタの設定
        )
    ]
    generated_texts = []

    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    for generate_config in generate_config_list:
        # テキスト生成
        output = model.generate(inputs, generation_config=generate_config)
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        # 読みやすくさの処理を行なって、リストに追加
        generated_texts.append("。\n".join(generated.replace(" ", "").split("。")))

    # gradioはtupleを想定している。これと同じ処理：return generated_texts[0], generated_texts[1], generated_texts[2]
    # pythonのタプルは「,」によって生成される。丸括弧は省略可能。参考：https://note.nkmk.me/python-function-return-multiple-values/
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
    """
    # デコード方法の指定に合わせて、cofingを定義
    if radio == "1.Greedy":
        generate_config = GenerationConfig(
            max_new_tokens=max_length,
            no_repeat_ngram_size=3,
            num_beams=1,
            do_sample=False
        )
    elif radio == "2.Sampling":
        generate_config = GenerationConfig(
            max_new_tokens=max_length,
            no_repeat_ngram_size=3,
            num_beams=1,
            do_sample=True
        )
    elif radio == "3.Beam Search":
        generate_config = GenerationConfig(
            max_new_tokens=max_length,
            no_repeat_ngram_size=3,
            num_beams=num_beams,
            do_sample=False
        )
    else:
        generate_config = GenerationConfig(
            max_new_tokens=max_length,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_p=p
        )

    # テキスト生成
    inputs = tokenizer(now_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    output = model.generate(inputs, generation_config=generate_config)
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    # 結果の整形処理
    next_text = "。\n".join(generated.replace(" ", "").split("。"))
    gen_text = next_text[len(now_text)+1:]  # 今回生成したテキストを抽出

    return next_text, next_text, gen_text

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

    with gr.Row():  # 行に分ける。なので、このブロック内にあるコンポーネントは横に並ぶ。
        with gr.Column():  # さらに列に分ける。なので、このブロック内にあるコンポーネントは縦に並ぶ。
            input_text = gr.Textbox(value="福岡のご飯は美味しい。", label="プロンプト")
            max_length = gr.Slider(100, 1000, step=100, value=100, label="生成するテキストの長さ")
            num_beams = gr.Slider(1, 10, step=1, value=6, label="beam幅")
            p = gr.Slider(0, 1, step=0.01, value=0.92, label="p")
            btn1 = gr.Button("４パターンで生成")
        
        with gr.Column():
            out1 = gr.Textbox(label="Greedy")
            out2 = gr.Textbox(label="Sampling")
            out3 = gr.Textbox(label="Beam Search")
            out4 = gr.Textbox(label="Top-p Sampling")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## どの結果の続きが気になりますか？")
            radio1 = gr.Radio(choices=["1.Greedy", "2.Sampling", "3.Beam Search", "4.Top-p Sampling"], value="1.Greedy", label="結果の選択")
            output_text = gr.Textbox(label="初回の結果")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"## どの方法で続きを生成しますか？")
            history = gr.State()
            now_text = gr.TextArea(label="これまでの結果")
            radio2 = gr.Radio(choices=["1.Greedy", "2.Sampling", "3.Beam Search", "4.Top-p Sampling"], value="1.Greedy", label="続き生成のデコード方法")
            btn2 = gr.Button("続きを生成")
            next_text = gr.TextArea(label="今回の生成結果")
            

    # 2.2 イベント処理
    btn1.click(fn=generate, inputs=[input_text, max_length, num_beams, p], outputs=[out1, out2, out3, out4])
    out1.change(select_out1, inputs=[out1], outputs=[output_text, history, now_text])
    radio1.change(select_out, inputs=[radio1, out1, out2, out3, out4], outputs=[output_text, history, now_text])
    btn2.click(fn=generate_next, inputs=[history, radio2, max_length, num_beams, p], outputs=[history, now_text, next_text])

if __name__ == "__main__":
    demo.launch()
