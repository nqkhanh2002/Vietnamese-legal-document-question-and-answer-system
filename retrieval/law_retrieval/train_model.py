import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification

from retrieval.constant import *
from src.train_cross_encoder.create_data import *

from sklearn.metrics import precision_recall_fscore_support

class MultilingualBertDataset(torch.utils.data.Dataset):
   
    def __init__(self, questions, articles, labels):
        self.questions = questions
        self.articles = articles
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def __len__(self):
        return len(self.labels)

    def tokenize_pair_text(self, text_1, text_2):
        return self.tokenizer(text_1, text_2, padding='max_length', truncation=True)

    def __getitem__(self, index):
        encodings = self.tokenize_pair_text(self.questions[index], self.articles[index])
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item

def f2_metric(preds, labels, num):
    f2 = 0
    len_sample = 0
    for i in range(0, len(labels), num):
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        len_sample += 1
        for j in range(i, i + num):
            local_label = int(labels[j])
            local_preds = int(preds[j])
            
            if local_label == 0:
                if local_preds == 0:
                    tn += 1
                else: fp += 1
            elif local_label == 1:
                if local_preds == 1:
                    tp += 1
                else: fn += 1
        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            local_f2 = (5 * precision * recall) / (4 * precision + recall)
        except:
            local_f2 = 0
        f2 += local_f2
    return f2/len_sample

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    f2 = (5 * precision * recall) / (4 * precision + recall)
    return {
        'precision': precision,
        'recall' : recall,
        'f2': f2,
    }

sigmoid_model = torch.nn.Sigmoid()
softmax_model = nn.Softmax(dim=1)


TRAIN_DATA = create_multi_case(TRAINING_PATH, CORPUS_PATH, NEGATIVE_MODE, NEGATIVE_NUM, 'train')
TEST_DATA = create_multi_case(TESTING_PATH, CORPUS_PATH, NEGATIVE_MODE, NEGATIVE_NUM, 'test')

if FAST_DEV_RUN == 1:
    TRAIN_DATA = {"question": TRAIN_DATA["question"][:100], "article": TRAIN_DATA["article"][:100], "label": TRAIN_DATA["label"][:100]}
    TEST_DATA = {"question": TEST_DATA["question"][:100], "article": TEST_DATA["article"][:100], "label": TEST_DATA["label"][:100]}

train_dataset = MultilingualBertDataset(TRAIN_DATA["question"], TRAIN_DATA["article"], TRAIN_DATA["label"])
test_dataset = MultilingualBertDataset(TEST_DATA["question"], TEST_DATA["article"], TEST_DATA["label"])

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
model = BertForSequenceClassification.from_pretrained(PRETRAIN_MODEL)
model.to(DEVICE)

if FREEZE_MODE == 1:
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.bert.embeddings.parameters():
        param.requires_grad = True

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        label_rate = [1.0, float(W_LOSS)]
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = softmax_model(outputs.get("logits"))
        loss_fct = nn.CrossEntropyLoss(weight=(torch.tensor(label_rate)).to(DEVICE))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

if OPTIMIZER == "SGD":    
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        num_train_epochs = N_EPOCH,
        per_device_train_batch_size = BATCH_SIZE,  
        per_device_eval_batch_size= BATCH_SIZE,
        learning_rate = LEARNING_RATE,
        evaluation_strategy = "epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        disable_tqdm = False, 
        load_best_model_at_end=True,
        save_total_limit=5
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        optimizers = (torch.optim.SGD(model.parameters(), lr=LEARNING_RATE), None)
    )
    print("SGD")
else:
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        num_train_epochs = N_EPOCH,
        per_device_train_batch_size = BATCH_SIZE,  
        per_device_eval_batch_size= BATCH_SIZE,
        learning_rate = LEARNING_RATE,
        evaluation_strategy = "epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        disable_tqdm = False, 
        load_best_model_at_end=True,
        save_total_limit=5
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    print("Adam")

trainer.train()