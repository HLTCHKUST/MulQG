
from GPG.trainer import Trainer
from GPG.inference import BeamSearcher
from GPG.config import set_config
import torch
from GPG.data.feature import *


def main():
    args = set_config()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    if not args.notrain: # train
        trainer = Trainer(args)
        trainer.train()
    else:
        assert args.restore is not None
        bs = BeamSearcher(args)
        score = bs.decode()
        print("the evaluation bleu score is {} \n".format(score))

if __name__ == "__main__":
    main()