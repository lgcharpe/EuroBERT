from collections import Counter
from tokenizers import Tokenizer


def bytes_to_string(b):
    return b.decode("utf-8", errors="replace")


if __name__ == '__main__':
    tokenizer_path = "../Australis_full/tokenizer.json"
    tokenizer_name = "Australis_full"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    quick_examples = [
    "it costs 12 dollars and 50 cents",
    "The number is 1234567890",
    "tokenize the word watermelon and pineapple",
    "supercalifragilisticexpialidocious",
    "ya man, we'd love to do that!!!"
    ]

    for example in quick_examples:
        encoding = tokenizer.encode(example)
        ids = encoding.ids
        tokens = encoding.tokens
        print(f"Example sentence: {example}")
        print(f"Tokenized sentence: {' | '.join(tokens)}\n")

    languages = ["English", "Danish", "Faroese", "Icelandic", "Bokmaal", "Nynorsk", "Sami", "Swedish", "Finnish", "Skolt Sami"]
    paths = [
    "UD_data/en_ewt-ud-train.conllu",
    "UD_data/da_ddt-ud-train.conllu",
    "UD_data/fo_farpahc-ud-train.conllu",
    "UD_data/is_modern-ud-train.conllu",
    "UD_data/no_bokmaal-ud-train.conllu",
    "UD_data/no_nynorsk-ud-train.conllu",
    "UD_data/sme_giella-ud-train.conllu",
    "UD_data/sv_lines-ud-train.conllu",
    "UD_data/fi_tdt-ud-train.conllu",
    "UD_data/sms_giellagas-ud-train.conllu"
    ]

    with open(f"{tokenizer_name}_fertility.txt", "w", encoding="utf-8") as out_file:
        for lang, validation_path in zip(languages, paths):
            print(f"========= Language {lang} =========", file=out_file)
            with open(validation_path, "r", encoding="utf-8") as f:
                text = f.read()

            sentences = text.split("\n\n")
            sentence_texts, sentence_lengths = [], []
            for sentence in sentences:
                if "# text = " not in sentence:
                    continue
                lines = sentence.split("\n")
                sentence_texts.append(sentence.split("# text = ")[1].split("\n")[0].strip())
                sentence_lengths.append(len([line for line in lines if not line.startswith("#") and line.split("\t")[0].isdigit()]))

            total_word_length, total_token_length = 0, 0
            counter = Counter()
            for i, (sentence_text, sentence_length) in enumerate(zip(sentence_texts, sentence_lengths)):
                encoding = tokenizer.encode(sentence_text, add_special_tokens=False)
                ids = encoding.ids
                tokens = encoding.tokens
                counter.update(tokens)
                total_word_length += sentence_length
                total_token_length += len(ids)

                if i % 1000 == 0:
                    print(f"Example sentence: {sentence_text}", file=out_file)
                    print(f"Tokenized sentence: {' | '.join(tokens)}\n", file=out_file)

            print(f"Most common tokens: {counter.most_common(10)}", file=out_file)

            print(f"Expected tokens per word: {total_token_length / total_word_length}", file=out_file)
