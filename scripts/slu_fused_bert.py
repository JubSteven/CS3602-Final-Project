# coding=utf8
import sys, os, time, gc, json
from torch.optim import Adam
from tqdm import tqdm
import json

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_bert_base import SLUFusedBertTagging
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.tensorBoard import visualizer

time_stramp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])

root_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_save_path = os.path.join(root_path, "checkpoints", args.expri + '_' + time_stramp)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path, exist_ok=True)

if args.expri == "empty":
    print("the name of this experiment is required for clarity!")
    exit(0)

print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")
set_random_seed(args.seed)
writer = visualizer(args)  # tensorboard writer

if args.device == -1:
    args.device = "cpu"
    device = "cpu"
else:
    device = set_torch_device(args.device)

start_time = time.time()
train_path = os.path.join(args.dataroot, 'CAIS_train.json')
# train_path = os.path.join(args.dataroot, 'train_augmented.json')
dev_path = os.path.join(args.dataroot, 'CAIS_test.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path, args)
dev_dataset = Example.load_dataset(dev_path, args)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

print("device", device)
model = SLUFusedBertTagging(args).to(device)
if args.encoder_cell == "naive-transformer" or args.encoder_cell == "naive-bert":
    Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)

if args.testing:
    check_point = torch.load(open(args.ckpt, 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    print("Load saved model from args.ckpt")


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice, wrong_examples_tag=None):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels, utts = [], [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        wrong_examples = dict()
        for i in range(0, len(dataset), args.batch_size):
            batch_labels = []
            batch_preds = []
            cur_dataset = dataset[i:i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            predictions.extend(pred)
            labels.extend(label)
            batch_preds.extend(pred)
            batch_labels.extend(label)
            total_loss += loss
            count += 1

            if wrong_examples_tag:
                wrong_examples[i // args.batch_size] = []
                for k in range(len(current_batch.utt)):
                    sentence = current_batch.utt[k]
                    pred = batch_preds[k]
                    label = batch_labels[k]

                    if set(pred) != set(label):
                        example = {"sentence": sentence, "pred": pred, "label": label}
                        wrong_examples[i // args.batch_size].append(example)
        if wrong_examples_tag:
            save_path = os.path.join(root_path, "wrong_examples", args.expri + '_' + time_stramp)
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f"{wrong_examples_tag}_wrong_examples.json"), 'w',
                      encoding='utf-8') as file:
                json.dump(wrong_examples, file, ensure_ascii=False, indent=4)
        metrics = Example.evaluator.acc(
            predictions, labels
        )  # here predictions and labels all comoposed of act-slot-value(maybe without slot or slot-value), for comparison

    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


def predict():
    model.eval()
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    test_dataset = Example.load_dataset(test_path, args)
    predictions = {}
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i:i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=False)
            pred = model.decode(Example.label_vocab, current_batch)
            for pi, p in enumerate(pred):
                did = current_batch.did[pi]
                predictions[did] = p
    test_json = json.load(open(test_path, 'r'))
    ptr = 0
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt['pred'] = [pred.split('-') for pred in predictions[f"{ei}-{ui}"]]
            ptr += 1
    json.dump(test_json, open(os.path.join(args.dataroot, 'prediction.json'), 'w'), indent=4, ensure_ascii=False)


if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    optimizer = set_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, batch_size = np.arange(nsamples), args.batch_size
    print('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index)
        model.train()
        count = 0
        trainbar = tqdm(range(0, nsamples, batch_size))
        for j, _ in enumerate(trainbar):
            cur_dataset = [train_dataset[k] for k in train_index[j:j + batch_size]]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            output, loss = model(current_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1

            if j % 50 == 0:
                msg = f"epoch={i}_batch={j}" if j == 0 and i % 5 == 0 else None
                metrics, dev_loss = decode('dev', msg)
                dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
                if dev_acc > best_result['dev_acc']:
                    best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result[
                        'iter'] = dev_loss, dev_acc, dev_fscore, i
                    model_name = f"gamma={args.gamma}_decay={args.decay_step}_lr={args.lr}.bin"
                    torch.save(
                        {
                            'epoch': i,
                            'model': model.state_dict(),
                            'optim': optimizer.state_dict(),
                            'results': best_result,
                        }, os.path.join(model_save_path, model_name))

                    print(
                        'NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)'
                        % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

                dev_info = {
                    "Best_Acc": best_result["dev_acc"],
                    "Dev_Acc": dev_acc,
                    "Dev_P": dev_fscore['precision'],
                    "Dev_R": dev_fscore['recall'],
                    "Dev_F": dev_fscore['fscore']
                }

                for key, value in dev_info.items():
                    writer.add_scalar(f"dev/{key}", value, j + i * 160)  # 160 = 「(nsamples / batch_size)

            trainbar.set_description(
                f"Epoch: {i} | L: {epoch_loss / count:.2f}| Best_Acc: {best_result['dev_acc']:.2f} | Acc: {dev_acc:.2f} | P: {dev_fscore['precision']:.2f} | R: {dev_fscore['recall']:.2f}| F: {dev_fscore['fscore']:.2f}"
            )
            writer.add_scalar("train/epoch_loss", epoch_loss / count, j + i * 160)  # 160 = 「(nsamples / batch_size)

        scheduler.step(dev_loss)
        torch.cuda.empty_cache()
        gc.collect()

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' %
          (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'],
           best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    predict()
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" %
          (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'],
           dev_fscore['fscore']))
