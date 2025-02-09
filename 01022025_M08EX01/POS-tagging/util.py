from sklearn.model_selection import train_test_split
import nltk
from dataset import PosTagging_Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification


def load_dataset():
    nltk.download("treebank")
    # Load tree bank dataset
    tagged_sentences = nltk.corpus.treebank.tagged_sents()
    print("Number of sentences:", len(tagged_sentences))

    # Save senteces and tages
    sentences, sentence_tags = [], []
    for tagged_sentence in tagged_sentences:
        sentence, tags = zip(*tagged_sentence)
        sentences.append([word.lower() for word in sentence])
        sentence_tags.append([tag for tag in tags])

    return sentences, sentence_tags


def preprocess_data(sentences, sentence_tags):
    train_sentences, test_sentences, train_tags, test_tags = train_test_split(
        sentences, sentence_tags, test_size=0.3
    )

    valid_sentences, test_sentences, valid_tags, test_tags = train_test_split(
        test_sentences, test_tags, test_size=0.5
    )

    return (
        train_sentences,
        valid_sentences,
        test_sentences,
        train_tags,
        valid_tags,
        test_tags,
    )


def dataset_loader(
    train_sentences,
    train_tags,
    valid_sentences,
    valid_tags,
    test_sentences,
    test_tags,
    tokenizer,
    label2id,
):
    train_dataset = PosTagging_Dataset(train_sentences, train_tags, tokenizer, label2id)
    valid_sentences = PosTagging_Dataset(
        valid_sentences, valid_tags, tokenizer, label2id
    )
    test_sentences = PosTagging_Dataset(test_sentences, test_tags, tokenizer, label2id)
    return train_dataset, valid_sentences, test_sentences


def modelling():
    model_name = "QCRI/bert-base-multilingual-cased-pos-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    return tokenizer, model
