from json_helper import JsonHelper
from geometry_utils import box_contains, find_closest_box
from document_processor import DocumentProcessor

PROMPT_TEMPLATE = """Below is the document:
$$$document$$$
{DocLayPrompt}
$$$/document$$$

$$$question$$$
{Question}
$$$/question$$$

$$$answer_format$$$
#Think step by step:xxx.
>Initial Variables:xxx,xxx.
>Si:function(input)->output;#exp:xxx.
>Si+1:function(input)->output;#exp:xxx.
>Si+2:...
#Final Answer:xxx
$$$/answer_format$$$

$$$task$$$
In the above document, the <xxx> represents the layout type, the original layout was attempted to be restored via insertion of spaces and newlines.
You need to use the pseudo-code function to analyze the answer step by step according to the document and question.
Your response have to strictly follow the answer_format.
Please response in English strictly. 
Function name should be as atomic as possible, special values must not appear in pseudo-code function but should appear in the explanation.
The input and output variables of the function should be traceable.
Only the final answer is given in the Final Answer.
$$$/task$$$

$$$example$$$
Question: What is the text under the title?
Answer:
#Think step by step:The question is asked to the text under title of document, we can determine the document title in the top of document fistly, then we can find the text under the title, so the answer is hi world.
>Initial Variables:document,title.
>S1:extract_layout_from_document(document,title)->title_text;#exp:find the title fo document, the title_text is "HELLO WORLD".
>S2:get_under_text(document,title_text)->under_title_text;#exp:extract the text under the title which is "hi world".
#Final Answer:"hi world".
$$$/example$$$

Your answer:
"""

def main():
    #export OPENAI_API_KEY="xxxxxx"
    model_name = "bigcode/gpt_bigcode-santacoder"
    tokenizer_name = "bigcode/gpt_bigcode-santacoder"

    clustering_params = {
        "eps": 0.03,
        "min_samples": 1,
        "metric": "cosine"
    }

    processor = DocumentProcessor()

    data = JsonHelper.load_json("data_examples/example.json")

    data = processor.generate_DocLayPrompt(data)
    
    data = processor.generate_proctags(data, PROMPT_TEMPLATE)

    data = processor.parse_proctags(data)

    data = processor.cluster_and_tag_procedures(data, model_name, tokenizer_name, clustering_params)

    sub_data = processor.complexity_first_diverse_sampling(data, 50)
    
    JsonHelper.save_json("data_examples/example_sub.json", sub_data)

if __name__ == "__main__":
    main()
