"""プロンプト生成(eval/llm_judge.py)のユニットテスト。Geminiへの実呼び出しはしない。"""

from llm_judge import build_answer_relevancy_prompt, build_faithfulness_prompt


def test_build_faithfulness_prompt_includes_question_and_sources():
    prompt = build_faithfulness_prompt(
        "副業は可能ですか？",
        "はい、事前届出により可能です。",
        [
            {
                "filename": "employment_rules.md",
                "heading_path": ["第9条"],
                "content": "副業は事前届出制。",
            }
        ],
    )
    assert "副業は可能ですか？" in prompt
    assert "employment_rules.md" in prompt
    assert "副業は事前届出制。" in prompt


def test_build_faithfulness_prompt_handles_no_sources():
    prompt = build_faithfulness_prompt("資本金は？", "情報がありません。", [])
    assert "検索結果なし" in prompt


def test_build_answer_relevancy_prompt_includes_reference_answer():
    prompt = build_answer_relevancy_prompt(
        "有給の繰越上限は？", "20日です。", "20日まで繰り越し可能。"
    )
    assert "有給の繰越上限は？" in prompt
    assert "20日まで繰り越し可能。" in prompt
