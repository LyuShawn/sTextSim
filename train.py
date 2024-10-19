from main.trainers.cse_trainer import Trainer as Trainer
from main.trainers.gcse_trainer import Trainer as GCSETrainer
# from main.trainers.glm_cse_trainer import Trainer as GLMCSETrainer
# from main.trainers.sts_trainer import Trainer as GLMGCSETrainer
from transformers import BertTokenizer,HfArgumentParser,set_seed
from dataclasses import dataclass, field
from typing import Optional
from main.auto_eval import eval

@dataclass
class TrainArguments:

    model_name_or_path: Optional[str] = field(default='bert-base-uncased',metadata={"help": "The model checkpoint for weights initialization."})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    data_present_path: Optional[str] = field(default='./dataset/present.json',metadata={"help": "The path of present data"})
    max_seq_len: Optional[int] = field(default=32, metadata={"help": "The max sequence length of input"})
    hard_negative_weight: Optional[float] = field(default=0, metadata={"help": "The weight of hard negative samples"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "The batch size of training"})
    temp: Optional[float] = field(default=0.05, metadata={"help": "The temperature of contrastive loss"})
    data_name: Optional[str] = field(default='WikiSTS', metadata={"help": "The name of dataset"})
    task_name: Optional[str] = field(default='SimCSE_Wiki_unsup', metadata={"help": "The name of task"})
    epochs: Optional[int] = field(default=1, metadata={"help": "The number of epochs for training"})
    lr : Optional[float] = field(default=3e-5, metadata={"help": "The learning rate for training"})
    eval_steps: Optional[int] = field(default=125, metadata={"help": "The steps for evaluation"})
    seed : Optional[int] = field(default=42, metadata={"help": "The seed for training"})
    do_train: Optional[bool] = field(default=False, metadata={"help": "Whether to do training"})
    do_eval: Optional[bool] = field(default=False, metadata={"help": "Whether to do evaluation"})

@dataclass
class ModelArguments:

    dropout: Optional[float] = field(default=0.1, metadata={"help": "The dropout rate of model"})


def main():

    model_base_path = './model/'
    save_path = './save_model/'

    parser = HfArgumentParser((TrainArguments, ModelArguments))
    train_args, model_args= parser.parse_args_into_dataclasses()

    if  '/' not in train_args.model_name_or_path:
        train_args.model_name_or_path = model_base_path + train_args.model_name_or_path
    if not train_args.tokenizer_name:
        train_args.tokenizer_name = train_args.model_name_or_path

    set_seed(train_args.seed)

    tokenizer = BertTokenizer.from_pretrained(train_args.model_name_or_path)
    trainer = Trainer(tokenizer=tokenizer,
                    from_pretrained=train_args.model_name_or_path,
                    data_present_path=train_args.data_present_path,
                    max_seq_len=train_args.max_seq_len,
                    hard_negative_weight=train_args.hard_negative_weight,
                    batch_size=train_args.batch_size,
                    temp=train_args.temp,
                    data_name=train_args.data_name,
                    task_name=train_args.task_name
                    dropout=model_args.dropout)

    if train_args.do_train:
        # do training
        for i in trainer(num_epochs=train_args.epochs, lr=train_args.lr, gpu=[0], eval_call_step=lambda x: x % train_args.eval_steps == 0):
            a = i

    if train_args.do_eval:
        # do evaluation
        best_model_path = save_path + train_args.task_name + '/simcse_best/'
        eval(best_model_path)


if __name__ == "__main__":
    main()