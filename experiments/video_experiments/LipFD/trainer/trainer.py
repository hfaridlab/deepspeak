import os
import torch
import torch.nn as nn
from models import build_model, get_loss


class Trainer(nn.Module):
    def __init__(self, opt):

        super(Trainer, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = (
            torch.device("cuda:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )
        self.opt = opt
        self.model = build_model(opt.arch)

        self.step_bias = (
            0
            if not opt.fine_tune
            else 100
        )
        if opt.fine_tune:
            state_dict = torch.load(opt.pretrained_model, map_location="cpu")
            self.model.load_state_dict(state_dict["model"])
            self.total_steps = state_dict["total_steps"]
            print(f"Model loaded @ {opt.pretrained_model.split('/')[-1]}")

        if opt.fix_encoder:
            params = []
            for name, p in self.model.named_parameters():
                if name.split(".")[0] in ["encoder"]:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    params.append(p)
            # params = self.model.parameters()

        if opt.optim == "adam":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.CrossEntropyLoss()

        self.model.to(opt.gpu_ids[0] if torch.cuda.is_available() else "cpu")

    def adjust_learning_rate(self, min_lr=1e-8):
        for param_group in self.optimizer.param_groups:
            if param_group["lr"] < min_lr:
                return False
            param_group["lr"] /= 10.0
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.crops = [[t.to(self.device) for t in sublist] for sublist in input[1]]
        self.label = input[2].to(self.device).float()

    def forward(self):
        self.get_features()
        self.output, self.weights_max, self.weights_org = self.model.forward(
            self.crops, self.features
        )
        self.output = self.output.view(-1)
        self.loss = self.criterion(
            self.weights_max, self.weights_org
        ) + self.criterion1(self.output, self.label)

    def get_loss(self):
        # loss = self.loss.data.tolist()
        # # if not loss.requires_grad:
        # #     print("Warning: Loss doesn't require grad!")
        # return loss[0] if isinstance(loss, type(list())) else loss
        if not self.loss.requires_grad:
            print("Warning: Loss doesn't require grad!")
        return self.loss.item()

    # def optimize_parameters(self):
    #     self.optimizer.zero_grad()
    #     self.loss.backward()
    #     self.optimizer.step()

    def optimize_parameters(self):
        if self.loss is None:
            print("Warning: Loss is None!")
            return
        if not self.loss.requires_grad:
            print("Warning: Loss doesn't require gradient!")
            return

        self.optimizer.zero_grad()

        loss_value = self.loss.item()

        self.loss.backward()

        total_grad_norm = 0
        num_params_with_grad = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    print(f"Warning: Parameter {name} has no gradient!")
                else:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    num_params_with_grad += 1

                    if torch.isnan(param.grad).any():
                        print(f"Warning: NaN gradients detected in {name}")
                    if torch.isinf(param.grad).any():
                        print(f"Warning: Inf gradients detected in {name}")

        self.optimizer.step()

        return {
            'loss': loss_value,
            'avg_grad_norm': total_grad_norm / num_params_with_grad if num_params_with_grad > 0 else 0
        }

    def get_features(self):
        self.features = self.model.get_features(self.input).to(
            self.device
        )  # shape: (batch_size

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

        torch.save(state_dict, save_path)