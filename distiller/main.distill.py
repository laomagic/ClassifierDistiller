import argparse
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger("Main")
import torch
import numpy as np
import random
from modeling import BertClassifier
from transformers import BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from data_processor import DictDataset
from utils import divide_parameters
from torch.optim import Adam
from textbrewer import TrainingConfig, DistillationConfig
from textbrewer import GeneralDistiller
from functools import partial
import textbrewer


def predict(model, step, args, eval_dataset, device):
    model.eval()
    pred_logits = []
    label_ids = []
    dataloader = DataLoader(eval_dataset, batch_size=64)
    logger.info("Predicting...")
    logger.info("***** Running predictions *****")
    logger.info("  Num  examples = %d", len(eval_dataset))
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels']
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            cpu_logits = outputs[0].detach().cpu()
        for i in range(len(cpu_logits)):
            pred_logits.append(cpu_logits[i].numpy())
            label_ids.append(labels[i])
    pred_logits = np.array(pred_logits)
    label_ids = np.array(label_ids)
    y_p = pred_logits.argmax(axis=-1)
    accuracy = (y_p == label_ids).sum() / len(label_ids)
    logger.info(f"accuracy:{accuracy}")
    model.train()
    return {"accuracy": accuracy}


def simple_adaptor(batch, model_outputs, no_logits, no_mask):
    dict_obj = {'hidden': model_outputs[2],
                'attention': model_outputs[3]}
    if no_mask is False:
        dict_obj['inputs_mask'] = batch['attention_mask']
    if no_logits is False:
        dict_obj['logits'] = (model_outputs[0])
    return dict_obj


def main(args):
    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")
    for k, v in vars(args).items():
        logger.info(f"{k}:{v}")
    # set  seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # read dataset
    tokenizer = BertTokenizer(vocab_file=args.vocab_file)
    train_dataset = DictDataset(tokenizer, args.train_dir, max_length=args.max_seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    num_labels = train_dataset.num_labels
    forward_batch_size = int(args.batch_size / args.gradient_accumulation_steps)
    num_train_steps = int(len(train_dataset) / args.batch_size) * args.num_epochs
    eval_dataset = DictDataset(tokenizer, args.eval_dir, max_length=args.max_seq_length)

    # bert_config
    bert_config_T = BertConfig.from_json_file(args.bert_config_file_T)
    bert_config_S = BertConfig.from_json_file(args.bert_config_file_S)
    assert args.max_seq_length <= bert_config_T.max_position_embeddings
    assert args.max_seq_length <= bert_config_S.max_position_embeddings

    # build model and load checkpoint
    model_T = BertClassifier(bert_config_T, num_labels=num_labels, args=args)
    model_S = BertClassifier(bert_config_S, num_labels=num_labels, args=args)
    # Load teacher
    if args.tuned_checkpoint_T is not None:
        state_dict_T = torch.load(args.tuned_checkpoint_T, map_location='cpu')
        model_T.load_state_dict(state_dict_T)
        model_T.eval()
    else:
        assert args.do_predict is True

    # load student
    if args.load_model_type == 'bert':
        assert args.init_checkpoint_S is not None
        state_dict_S = torch.load(args.init_checkpoint_S, map_location='cpu')
        if args.only_load_embedding:
            state_weight = {k[5:]: v for k, v in state_dict_S.items() if k.startswith('bert.embeddings')}
            missing_keys, _ = model_S.bert.load_state_dict(state_weight, strict=False)
            logger.info(f"Missing keys {list(missing_keys)}")
        else:
            # state_weight = {k[5:]: v for k, v in state_dict_S.items() if
            #                 k.startswith('bert.encoder') or k.startswith('bert.pooler')}
            state_weight = {k[5:]: v for k, v in state_dict_S.items() if k.startswith('bert.')}
            missing_keys, _ = model_S.bert.load_state_dict(state_weight, strict=False)
            # assert len(missing_keys) == 0
        logger.info("Model loaded")
    elif args.load_model_type == 'all':
        assert args.tuned_checkpoint_S is not None
        state_dict_S = torch.load(args.tuned_checkpoint_S, map_location='cpu')
        model_S.load_state_dict(state_dict_S)
        logger.info("Model loaded")
    else:
        logger.info("Model is randomly initialized.")

    # display model parameters statistics
    logger.info("\nteacher_model's parametrers:")
    teacher_result, _ = textbrewer.utils.display_parameters(model_T, max_level=3)
    logger.info(f'{teacher_result}')

    logger.info("student_model's parametrers:")
    student_result, _ = textbrewer.utils.display_parameters(model_S, max_level=3)
    logger.info(f'{student_result}')

    model_T.to(device)
    model_S.to(device)

    if args.do_train:
        params = list(model_S.named_parameters())
        all_trainable_params = divide_parameters(params, lr=args.learning_rate)
        logger.info("Length of all_trainable_params: %d", len(all_trainable_params))

        optimizer = Adam(all_trainable_params, lr=args.learning_rate)
        scheduler_class = get_linear_schedule_with_warmup
        scheduler_args = {'num_warmup_steps': int(0.1 * num_train_steps), 'num_training_steps': num_train_steps}

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Forward batch size = %d", forward_batch_size)
        logger.info("  Num backward steps = %d", num_train_steps)

        ########### DISTILLATION ###########
        train_config = TrainingConfig(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_dir=args.log_dir,
            output_dir=args.output_dir,
            device=device)
        from matches import matches
        intermediate_matches = []
        #  t3 L3_hidden_mse + L3_hidden_smmd
        #  t3-small L3n_hidden_mse + L3_hidden_smmd
        # for match in ["L3_hidden_mse", "L3_hidden_smmd"]:
        for match in args.matches:
            intermediate_matches += matches[match]
        logger.info(f"{intermediate_matches}")

        distill_config = DistillationConfig(
            temperature=args.temperature,
            hard_label_weight=0,
            kd_loss_type='ce',
            probability_shift=False,
            intermediate_matches=intermediate_matches)

        logger.info(f"{train_config}")
        logger.info(f"{distill_config}")

        adaptor_T = partial(simple_adaptor, no_logits=False, no_mask=False)
        adaptor_S = partial(simple_adaptor, no_logits=False, no_mask=False)

        distiller = GeneralDistiller(train_config=train_config,
                                     distill_config=distill_config,
                                     model_T=model_T, model_S=model_S,
                                     adaptor_T=adaptor_T,
                                     adaptor_S=adaptor_S)

        callback_func = partial(predict, eval_dataset=eval_dataset, args=args, device=device)
        with distiller:
            distiller.train(optimizer, scheduler=None,
                            dataloader=train_dataloader,
                            num_epochs=args.num_epochs,
                            scheduler_class=scheduler_class,
                            scheduler_args=scheduler_args,
                            callback=callback_func)

    if not args.do_train and args.do_predict:
        res = predict(model_S, 1, args, eval_dataset, device)
        print(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model distill")
    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument("--vocab_file", default='model/roberta/vocab.txt', type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument('--train_dir', dest="train_dir", action="store", default='../atec_nlp_sim_train_all.csv',
                        help="train file")
    parser.add_argument('--eval_dir', dest="eval_dir", action="store", default='../dev.csv', help="eval file")
    parser.add_argument("--log_dir", dest="log_dir", action="store", default="logs", help="")
    parser.add_argument("--output_dir", dest="output_dir", action="store", default="save_dir", help="")

    parser.add_argument("--batch_size", dest="batch_size", action="store", type=int, default=64, help="")
    parser.add_argument("--max_seq_length", dest="max_seq_length", action="store", type=int, default=103, help="")
    parser.add_argument("--temperature", dest="temperature", action="store", type=int, default=8, help="")
    parser.add_argument("--num_epochs", dest="num_epochs", action="store", type=int, default=40, help="")
    parser.add_argument("--learning_rate", dest="learning_rate", action="store", type=float, default=0.0001, help="")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument('--schedule', type=str, default='warmup_linear_release')
    parser.add_argument("--bert_config_file_T", default='config/bert_config.json', type=str, required=True)
    parser.add_argument("--bert_config_file_S", default='config/bert_config_L3.json', type=str, required=True)
    parser.add_argument('--tuned_checkpoint_T', type=str, default=None)
    parser.add_argument("--init_checkpoint_S", default=None, type=str)
    parser.add_argument('--only_load_embedding', action='store_true', default=False)
    parser.add_argument('--load_model_type', type=str, default='none', choices=['bert', 'all', 'none'])
    parser.add_argument('--output_hidden_states', default='true', choices=['true', 'false'])
    parser.add_argument('--output_attentions', default='true', choices=['true', 'false'])
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", action="store",
                        type=int, default=4, help="")
    parser.add_argument('--matches', nargs='*', type=str)
    args = parser.parse_args()
    main(args)
