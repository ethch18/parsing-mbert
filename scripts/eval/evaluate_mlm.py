import argparse
import transformers

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--data', type=str)
args = parser.parse_args()

tokenizer = transformers.BertTokenizer(
        vocab_file=args.vocab, do_lower_case=False, do_basic_tokenize=True) 
model = transformers.BertForMaskedLM.from_pretrained(args.model)

dataset = transformers.LineByLineTextDataset(
        tokenizer=tokenizer, file_path=args.data, block_size=128)
data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
train_args = transformers.TrainingArguments(
        per_device_eval_batch_size=16, output_dir=f"/tmp/echau18/{args.model}")
trainer = transformers.Trainer(
        model=model, eval_dataset=dataset, data_collator=data_collator,
        prediction_loss_only=True, args=train_args)

eval_output = trainer.evaluate()
print(eval_output)

